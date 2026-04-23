import os
import sys
import time
import torch
import csv
import argparse
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from copy import deepcopy

from src.args import parse_arguments
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder, ImageClassifier, ClassificationHead
from src.utils import cosine_lr, LabelSmoothing
from src.cl_utils import get_dataset_and_classifier_for_split
from src.online_utils import LossLandscapeDetector
from src.heads import build_classification_head
import torchvision.transforms as transforms


class OffsetDataset(torch.utils.data.Dataset):
    """
    複数データセットのラベル重複を防ぐため、ターゲットラベルに一定のオフセットを加算するラッパー
    """
    def __init__(self, subset, offset):
        self.subset = subset
        self.offset = offset
        
    def __getitem__(self, idx):
        item = self.subset[idx]
        if isinstance(item, dict):
            item['labels'] = item['labels'] + self.offset
            return item
        elif len(item) == 2:
            return {'images': item[0], 'labels': item[1] + self.offset}
        elif len(item) == 3:
            return {'images': item[0], 'labels': item[1] + self.offset, 'image_paths': item[2]}
        return item

    def __len__(self):
        return len(self.subset)


def train_online(args):
    # デバイス設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
        args.device = "cuda"
    else:
        device = torch.device("cpu")
        args.device = "cpu"
    
    print(f"Using device: {device}")

    # 複数データセットの対応 (カンマ区切りで指定された場合)
    datasets_list = args.dataset.split(',')
    is_multi_dataset = len(datasets_list) > 1
    dataset_dir_name = args.dataset.replace(',', '_')

    # コマンドラインから受け取った設定値を反映
    if is_multi_dataset:
        args.n_splits = args.num_cycles * len(datasets_list)

    # --- 1. 保存ディレクトリの準備 ---
    sequential_ft_dir = 'sequential_finetuning/' if args.sequential_finetuning else ''
    base_save_dir = f'checkpoints/{args.model}/{sequential_ft_dir}{args.split_strategy}_incremental'
    
    ckpdir = os.path.join(base_save_dir,
                          f"{dataset_dir_name}-tasks:{args.n_splits}-cls_per_task:{args.classes_per_task}",
                          f"ft-epochs-{args.epochs}-seed:{args.seed}-{args.timestamp}"
                          )
    os.makedirs(ckpdir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpdir}")
    log_csv_path = os.path.join(ckpdir, "loss_history.csv")
    
    # ---------------------------------------------

    # 2. データセットの準備
    print("Preparing Task-Free Data Stream...")
    print('Building image encoder.')
    image_encoder = ImageEncoder(args, keep_lang=True)
    preprocess_fn = transforms.Compose([
        image_encoder.train_preprocess,
        transforms.CenterCrop(224)
    ])
    
    stream_datasets = []
    
    if not is_multi_dataset:
        # ===== 従来の単一データセットの設定 =====
        full_dataset_obj = get_dataset(
            args.dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        
        for split_idx in range(args.n_splits):
            dataset_part, _ = get_dataset_and_classifier_for_split(
                deepcopy(full_dataset_obj), 
                split_idx, 
                image_encoder, 
                args, 
                return_classifier=True
            )
            stream_datasets.append(dataset_part.train_dataset)
            
        # 初期化用のHeadを取得
        _, init_head = get_dataset_and_classifier_for_split(
                deepcopy(full_dataset_obj), 0, image_encoder, args, return_classifier=True
            )
        
        # 共通の初期化Headを保存
        head_init_path = f'checkpoints/{args.model}/{args.dataset}_full_head_init.pt'
        if not os.path.exists(head_init_path):
            os.makedirs(os.path.dirname(head_init_path), exist_ok=True)
            init_head.save(head_init_path)

    else:
        # ===== 複数データセット混在の設定 =====
        full_dataset_objs = {}
        total_classes = 0
        class_offsets = {}
        all_classnames = []
        
        print(f"Multi-dataset mode active: {datasets_list}")
        print(f"Settings: {args.classes_per_task} classes per task, {args.num_cycles} cycles.")
        
        for ds_name in datasets_list:
            ds_obj = get_dataset(
                ds_name,
                preprocess_fn,
                location=args.data_location,
                batch_size=args.batch_size
            )
            full_dataset_objs[ds_name] = ds_obj
            class_offsets[ds_name] = total_classes
            total_classes += len(ds_obj.classnames)
            all_classnames.extend(ds_obj.classnames)
            
        print(f"Total classes across datasets: {total_classes}")
        
        # サイクルごとに各データセットから指定クラス数ずつ抽出して追加
        for cycle_idx in range(args.num_cycles):
            for ds_name in datasets_list:
                ds_obj = full_dataset_objs[ds_name]
                
                # そのデータセットを指定クラス数で分割した場合の仮想的な分割数
                virtual_n_splits = len(ds_obj.classnames) // args.classes_per_task
                
                # データセットが持つクラス数を超えてサイクルを回そうとした場合の安全対策
                if cycle_idx >= virtual_n_splits:
                    print(f"Warning: Cycle {cycle_idx} exceeds available classes for {ds_name}. Skipping this task.")
                    continue
                
                temp_args = deepcopy(args)
                temp_args.n_splits = virtual_n_splits
                
                dataset_part = get_dataset_and_classifier_for_split(
                    deepcopy(ds_obj), 
                    cycle_idx, 
                    image_encoder, 
                    temp_args, 
                    return_classifier=False
                )
                
                # ラベルが重複しないようにオフセットを加算
                offset = class_offsets[ds_name]
                offset_dataset = OffsetDataset(dataset_part.train_dataset, offset)
                stream_datasets.append(offset_dataset)
                
        # 複数データセット用の共通Head（統合サイズ）の初期化
        head_init_path = f'checkpoints/{args.model}/multi_{dataset_dir_name}_full_head_init.pt'
        if not os.path.exists(head_init_path):
            os.makedirs(os.path.dirname(head_init_path), exist_ok=True)
            init_head = build_classification_head(
                image_encoder.model, 
                "multi_dataset", 
                args.data_location, 
                args.device, 
                all_classnames
            )
            init_head.save(head_init_path)
        else:
            print(f"Loading common initialization head from {head_init_path}")
            try:
                init_head = ClassificationHead.load(head_init_path)
            except:
                init_head = torch.load(head_init_path)
            init_head = init_head.to(args.device)


    # DataLoaderの構築
    continuous_dataset = ConcatDataset(stream_datasets)
    stream_loader = DataLoader(
        continuous_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    print(f"Total Stream Length: {len(continuous_dataset)} samples")

    # 3. モデルの準備
    model = ImageClassifier(image_encoder, init_head)
    model.freeze_lang()
    
    # GPUへ転送
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=devices)
        model = model.cuda()
    else:
        model = model.to(device)

    # 損失関数とオプティマイザ
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available() and hasattr(loss_fn, 'to'):
        loss_fn = loss_fn.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    # 4. 検知器の初期化
    detector = LossLandscapeDetector(
        window_length=args.loss_window_length,
        slope_threshold=5e-2,  
        peak_factor=3.0,       
        min_task_duration=0   
    )

    # 5. オンライン学習ループ
    print("\nStart Online Training...")
    model.train()
    
    total_steps = len(stream_loader)
    detected_count = 0 

    history = []

    for step, batch in enumerate(stream_loader):
        batch = maybe_dictionarize(batch)
        
        if torch.cuda.is_available():
            inputs = batch['images'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)
        else:
            inputs = batch['images'].to(device)
            labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        # 検知ロジック
        current_loss_val = loss.item()
        is_plateau, is_peak = detector.update(current_loss_val)

        current_slope = detector.current_slope
        current_min_mean = detector.current_task_min_mean
        
        event_type = "None"
        if is_plateau: event_type = "Plateau"
        if is_peak: event_type = "Peak"
        
        history.append({
            "step": step,
            "loss": current_loss_val,
            "slope": current_slope,
            "min_mean": current_min_mean if current_min_mean != float('inf') else np.nan,
            "in_peak_region": int(detector.in_peak_region),
            "event": event_type
        })
        
        if step % 10 == 0:
            print(f"Step [{step}/{total_steps}] Loss: {current_loss_val:.4f} "
                  f"Slope: {current_slope:.6f} "
                  f"MinMean: {current_min_mean:.4f}", end="\r")
        
        if is_peak:
            print(f"\n[!] New Peak Detected at step {step} (Loss Increased)")

        if is_plateau:
            print(f"\n[!] Plateau Detected at step {step} (Count: {detected_count})")
            
            enc_path = os.path.join(ckpdir, f'finetuned_{detected_count}.pt')
            head_path = os.path.join(ckpdir, f'head_{detected_count}.pt')
            
            if isinstance(model, torch.nn.DataParallel):
                model.module.image_encoder.save(enc_path)
                model.module.classification_head.save(head_path)
            else:
                model.image_encoder.save(enc_path)
                model.classification_head.save(head_path)
            
            print(f"    -> Saved encoder to {enc_path}")
            print(f"    -> Saved head to {head_path}")
            
            detected_count += 1
            
            if torch.cuda.is_available():
                model = model.cuda()
            else:
                model = model.to(device)

    print("\nOnline Training Finished.")
    # 6. CSVへの保存
    print(f"Saving loss history to {log_csv_path} ...")
    keys = history[0].keys()
    with open(log_csv_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(history)
    print("Done.")


if __name__ == '__main__':
    # 元の parse_arguments() が未定義の引数でエラーを吐かないように、
    # 先に新しい引数だけを parse_known_args() で抜き出します。
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument('--classes_per_task', type=int, default=10, help='Number of classes per task in multi-dataset mode')
    custom_parser.add_argument('--num_cycles', type=int, default=3, help='Number of cycles across datasets')
    
    custom_args, remaining_argv = custom_parser.parse_known_args()
    
    # 残りの引数を元のパーサーに渡すために sys.argv を上書き
    sys.argv = [sys.argv[0]] + remaining_argv
    
    args = parse_arguments()
    
    # 取り出した独自の引数を args に追加
    args.classes_per_task = custom_args.classes_per_task
    args.num_cycles = custom_args.num_cycles
    
    args.batch_size = 16
    
    if not args.split_strategy:
        args.split_strategy = 'class'
    
    args.sequential_finetuning = True

    print('='*100)
    print(f'Online Task-Free Training: {args.model} on {args.dataset}')
    if ',' in args.dataset:
        print(f'Multi-Dataset Setup: {args.classes_per_task} classes/task, {args.num_cycles} cycles')
    print(f'Detection Thresholds -> Mean: {args.loss_window_mean_threshold}, Var: {args.loss_window_variance_threshold}')
    print('='*100)

    train_online(args)