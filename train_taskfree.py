import os
import time
import torch
import csv
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

    # --- 1. 保存ディレクトリの準備 (MagMax互換) ---
    # merge_splitted.py が読み込めるディレクトリ構造を作成します
    # 例: checkpoints/ViT-B-16/sequential_finetuning/class_incremental/CIFAR100-10/ft-epochs-1-seed:5-2025.../
    
    sequential_ft_dir = 'sequential_finetuning/' if args.sequential_finetuning else ''
    base_save_dir = f'checkpoints/{args.model}/{sequential_ft_dir}{args.split_strategy}_incremental'
    
    ckpdir = os.path.join(base_save_dir,
                          f"{args.dataset}-{args.n_splits}",
                          f"ft-epochs-{args.epochs}-seed:{args.seed}-{args.timestamp}"
                          )
    os.makedirs(ckpdir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpdir}")
    # ログ保存用CSVパス
    log_csv_path = os.path.join(ckpdir, "loss_history.csv")
    
    # ---------------------------------------------

    # 2. データセットの準備
    print("Preparing Task-Free Data Stream...")
    print('Building image encoder.')
    image_encoder = ImageEncoder(args, keep_lang=True)
    preprocess_fn = image_encoder.train_preprocess
    
    full_dataset_obj = get_dataset(
        args.dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    
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

    # 3. モデルの準備
    # 初期化用のHeadを取得
    _, init_head = get_dataset_and_classifier_for_split(
            deepcopy(full_dataset_obj), 0, image_encoder, args, return_classifier=True
        )
    
    # 共通の初期化Headを保存 (mergeスクリプトが参照する場合があるため)
    head_init_path = f'checkpoints/{args.model}/{args.dataset}_full_head_init.pt'
    if not os.path.exists(head_init_path):
        os.makedirs(os.path.dirname(head_init_path), exist_ok=True)
        init_head.save(head_init_path)

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
    #optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0)
    
    # 4. 検知器の初期化
    detector = LossLandscapeDetector(
        window_length=args.loss_window_length,
        slope_threshold=5e-4,  # 傾き閾値
        peak_factor=3.0,       # ピーク感度
        min_task_duration=0   # 最低タスク長
    )

    # 5. オンライン学習ループ
    print("\nStart Online Training...")
    model.train()
    
    total_steps = len(stream_loader)
    detected_count = 0 # 検知回数カウンタ (split_idxの代わり)

    #履歴保存用リスト
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

        # 現在の状態を取得
        current_slope = detector.current_slope
        current_min_mean = detector.current_task_min_mean
        
        # イベント情報を記録
        event_type = "None"
        if is_plateau: event_type = "Plateau"
        if is_peak: event_type = "Peak"
        
        # 履歴に追加
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
        
        # --- 新しいピーク（タスク開始）検知時のログ ---
        if is_peak:
            print(f"\n[!] New Peak Detected at step {step} (Loss Increased)")
            #print(f"    Loss Mean: {detector.last_loss_window_mean:.4f}, Var: {detector.last_loss_window_variance:.4f}")

        # --- Plateau検知時の保存処理 ---
        if is_plateau:
            print(f"\n[!] Plateau Detected at step {step} (Count: {detected_count})")
            #print(f"    Loss Mean: {detector.last_loss_window_mean:.4f}, Var: {detector.last_loss_window_variance:.4f}")
            
            # MagMax互換のファイル名で保存
            # finetuned_{idx}.pt (エンコーダー)
            # head_{idx}.pt (分類ヘッド)
            
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
            
            # モデルをGPUに戻す（save内でcpu()が呼ばれるため）
            if torch.cuda.is_available():
                model = model.cuda()
            else:
                model = model.to(device)

    print("\nOnline Training Finished.")
    # 5. CSVへの保存
    print(f"Saving loss history to {log_csv_path} ...")
    keys = history[0].keys()
    with open(log_csv_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(history)
    print("Done.")

if __name__ == '__main__':
    args = parse_arguments()
    args.batch_size = 16
    
    # 以下の引数が指定されていない場合のデフォルト動作を補正
    if not args.split_strategy:
        args.split_strategy = 'class'
    
    # Task-Freeは本質的にSequentialなのでTrueとして扱う（パス生成のため）
    args.sequential_finetuning = True

    print('='*100)
    print(f'Online Task-Free Training: {args.model} on {args.dataset}')
    print(f'Detection Thresholds -> Mean: {args.loss_window_mean_threshold}, Var: {args.loss_window_variance_threshold}')
    print('='*100)

    train_online(args)