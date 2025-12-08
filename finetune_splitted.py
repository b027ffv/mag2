import os
import time
import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.cl_utils import get_dataset_and_classifier_for_split
from src.merging.task_vectors import TaskVector
from src.eval import evaluate

def finetune(args):
    
    train_dataset = args.dataset
    # --- 変更: ディレクトリ名に timestamp を追加 ---
    ckpdir = os.path.join(args.save,
                          f"{train_dataset}-{args.n_splits}",
                          f"ft-epochs-{args.epochs}-seed:{args.seed}-{args.timestamp}" # 変更箇所
                          )
    # ---------------------------------------------
    # 各分割タスクの学習
    for split_idx in range(args.n_splits):
        
        print(f"\n##### SPLIT {split_idx} #####")
        ft_path = os.path.join(ckpdir, f'finetuned_{split_idx}.pt')
        if os.path.exists(ft_path):
            print(f"Skipping finetuning on split {split_idx}, ckpt already exists.")
            continue

        assert train_dataset is not None, "Please provide a training dataset."
        
        # エンコーダーの読み込み（変更なし）
        if args.load is not None and args.load.endswith('pt'):
            image_encoder = ImageEncoder.load(args.load, keep_lang=True)
        elif args.sequential_finetuning and split_idx != 0:
            prev_ckpt = os.path.join(ckpdir, f'finetuned_{split_idx-1}.pt')
            print(f'Loading image encoder from prev task {prev_ckpt=}')
            image_encoder = torch.load(prev_ckpt)
        else:
            print('Building image encoder.')
            image_encoder = ImageEncoder(args, keep_lang=True)

        if split_idx == 0 and not os.path.exists(f'checkpoints/{args.model}/zeroshot.pt'):
            image_encoder.save(f'checkpoints/{args.model}/zeroshot.pt')

        preprocess_fn = image_encoder.train_preprocess
        print_every = 100

        dataset = get_dataset(
            train_dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        
        # 修正版の関数を呼び出し（active_classesが付与されたdatasetが返る）
        dataset, classification_head = get_dataset_and_classifier_for_split(
            dataset, split_idx, image_encoder, args
        )
        
        model = ImageClassifier(image_encoder, classification_head)
        model.freeze_head()
        model.freeze_lang()

        devices = list(range(torch.cuda.device_count()))
        print('Using devices', devices)
        model = torch.nn.DataParallel(model, device_ids=devices)

        if args.ls > 0:
            loss_fn = LabelSmoothing(args.ls)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        num_batches = len(dataset.train_loader)
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
        n_batches = len(data_loader)

        if args.save is not None:
            os.makedirs(ckpdir, exist_ok=True)

        # 【重要】 Logit Mask の作成
        # 現在のタスクに関係ないクラスをマスクするためのテンソルを準備
        active_classes = dataset.active_classes
        if active_classes is None or len(active_classes) == 0:
            # Data incrementalなどの場合は全クラス対象
            use_mask = False
        else:
            use_mask = True
            print(f"Masking logits for non-active classes. Active: {len(active_classes)} classes.")

        for epoch in range(args.epochs):
            model = model.cuda()
            model.train()

            for i, batch in enumerate(data_loader):
                start_time = time.time()

                step = i + epoch * num_batches
                scheduler(step)
                optimizer.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to('cuda:0')
                labels = batch['labels'].to('cuda:0')
                data_time = time.time() - start_time

                # 順伝播
                logits = model(inputs)

                # 【変更点】 Logit Masking の適用
                if use_mask:
                    # 全てを -inf で初期化
                    mask = torch.full_like(logits, float('-inf'))
                    # アクティブなクラスだけ 0 にする（元の値を保持）
                    mask[:, active_classes] = 0
                    # マスクを適用（非アクティブクラスは -inf になり、Softmaxで0になる）
                    logits = logits + mask

                loss = loss_fn(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                batch_time = time.time() - start_time

                if step % print_every == 0 or i + 1 == n_batches:
                    percent_complete = 100 * i / len(data_loader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                    )

        # Evaluate
        # 評価時はマスクなし（全クラスでの評価）またはマスクありを選べますが、
        # 通常のCIL評価では、学習した範囲のクラスのみ、あるいは全クラスで評価します。
        # ここでは変更せずそのまま評価関数に渡します。
        image_encoder = model.module.image_encoder
        evaluate(image_encoder, args)

        if args.save is not None:
            # エンコーダーの保存
            image_encoder.save(ft_path)
            
            # 【重要】 Classification Head の保存
            # Task Vector等で使用するために、学習後のフルサイズHeadを保存します
            head_path = os.path.join(ckpdir, f'head_{split_idx}.pt')
            if isinstance(model, torch.nn.DataParallel):
                model.module.classification_head.save(head_path)
            else:
                model.classification_head.save(head_path)


if __name__ == '__main__':
    args = parse_arguments()

    args.lr = 2e-4
    args.batch_size = 16 # 必要に応じて調整
    sequential_ft_dir = 'sequential_finetuning/' if args.sequential_finetuning else ''
    args.save = f'checkpoints/{args.model}/{sequential_ft_dir}{args.split_strategy}_incremental'

    print('='*100)
    print(f'Finetuning {args.model} on {args.dataset} ({args.n_splits} splits) [Full-Head Init]')
    print('='*100)

    finetune(args)