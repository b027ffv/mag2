import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from copy import deepcopy

from src.args import parse_arguments
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.cl_utils import get_dataset_and_classifier_for_split
from src.online_utils import LossLandscapeDetector

def train_online(args):
    # デバイス設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
        args.device = "cuda"
    else:
        device = torch.device("cpu")
        args.device = "cpu"
    
    print(f"Using device: {device}")

    # 1. データセットの準備
    print("Preparing Task-Free Data Stream...")
    
    print('Building image encoder.')
    image_encoder = ImageEncoder(args, keep_lang=True)
    preprocess_fn = image_encoder.train_preprocess
    
    # データセット全体を一度ロードする
    full_dataset_obj = get_dataset(
        args.dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    
    # ストリーム用データローダー
    stream_datasets = []
    for split_idx in range(args.n_splits):
        dataset_part, _ = get_dataset_and_classifier_for_split(
            deepcopy(full_dataset_obj), 
            split_idx, 
            image_encoder, 
            args, 
            return_classifier=True
        )
        stream_datasets.append(dataset_part.train_dataset)
    
    continuous_dataset = ConcatDataset(stream_datasets)
    
    stream_loader = DataLoader(
        continuous_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Total Stream Length: {len(continuous_dataset)} samples")

    # 2. モデルとオプティマイザの準備
    _, init_head = get_dataset_and_classifier_for_split(
            deepcopy(full_dataset_obj), 0, image_encoder, args, return_classifier=True
        )
    
    # モデル構築
    model = ImageClassifier(image_encoder, init_head)
    model.freeze_lang()
    
    # GPUへ転送
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
        print('Using devices', devices)
        model = torch.nn.DataParallel(model, device_ids=devices)
        model = model.cuda()
    else:
        model = model.to(device)
    
    # デバッグ: パラメータのデバイス確認
    try:
        if isinstance(model, torch.nn.DataParallel):
            param = next(model.module.parameters())
        else:
            param = next(model.parameters())
        print(f"[Debug] Model first parameter is on: {param.device}")
    except StopIteration:
        pass

    # 損失関数
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available() and hasattr(loss_fn, 'to'):
        loss_fn = loss_fn.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    # 3. 検知器の初期化
    detector = LossLandscapeDetector(
        window_length=args.loss_window_length,
        mean_threshold=args.loss_window_mean_threshold,
        var_threshold=args.loss_window_variance_threshold
    )

    # 4. オンライン学習ループ
    print("\nStart Online Training...")
    model.train()
    
    total_steps = len(stream_loader)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"outs/{args.model}/task_free/{args.dataset}/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    for step, batch in enumerate(stream_loader):
        batch = maybe_dictionarize(batch)
        
        if torch.cuda.is_available():
            inputs = batch['images'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)
        else:
            inputs = batch['images'].to(device)
            labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        # 検知ロジック
        current_loss_val = loss.item()
        is_boundary = detector.update(current_loss_val)
        
        if step % 10 == 0:
            print(f"Step [{step}/{total_steps}] Loss: {current_loss_val:.4f} "
                  f"Mean: {detector.last_loss_window_mean:.4f} "
                  f"Var: {detector.last_loss_window_variance:.4f}", end="\r")

        # --- タスク境界検知時の処理 ---
        if is_boundary:
            print(f"\n[!] Task Boundary Detected at step {step}!")
            print(f"    Loss Mean: {detector.last_loss_window_mean:.4f}, Var: {detector.last_loss_window_variance:.4f}")
            
            ckpt_path = os.path.join(out_dir, f"checkpoint_detected_step_{step}.pt")
            
            # 保存 (この処理内で model.cpu() が呼ばれる)
            if isinstance(model, torch.nn.DataParallel):
                model.module.save(ckpt_path)
            else:
                model.save(ckpt_path)
            print(f"    -> Checkpoint saved to {ckpt_path}")

            # 【修正箇所】保存によってCPUに移動したモデルをGPUに戻す
            if torch.cuda.is_available():
                model = model.cuda()
            else:
                model = model.to(device)

    print("\nOnline Training Finished.")

if __name__ == '__main__':
    args = parse_arguments()
    args.batch_size = 16
    
    print('='*100)
    print(f'Online Task-Free Training: {args.model} on {args.dataset}')
    if torch.cuda.is_available():
        print(f'Device: CUDA ({torch.cuda.get_device_name(0)})')
    else:
        print('Device: CPU')
    print('='*100)

    train_online(args)