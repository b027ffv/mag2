import torch
import os
from torch.utils.data.dataset import Subset

from src.datasets.common import (
    get_balanced_data_incremental_subset_indices,
    get_class_incremental_classes_and_subset_indices,
)
from src.heads import get_classification_head, build_classification_head, ClassificationHead
from src.modeling import ImageEncoder

def get_dataset_and_classifier_for_split(dataset, split_idx, text_encoder, args, remap_labels=True, return_classifier=True):
    # アクティブなクラス（現在のタスクで学習するクラス）を保存するリスト
    active_classes = []

    if args.split_strategy == 'data':
        train_subset_indices, test_subset_indices = \
            get_balanced_data_incremental_subset_indices(
                dataset.train_dataset, args.n_splits, split_idx
            )
        dataset.train_dataset = torch.utils.data.Subset(dataset.train_dataset, train_subset_indices)
        if return_classifier:
            classification_head = get_classification_head(args, args.dataset)
            
    elif args.split_strategy == 'class':
        classes, train_subset_indices, test_subset_indices = \
            get_class_incremental_classes_and_subset_indices(
                dataset, args.n_splits, split_idx
            )
        
        # 現在のタスクのアクティブクラスを記録
        active_classes = sorted(classes)

        dataset.train_dataset = Subset(dataset.train_dataset, train_subset_indices)
        dataset.test_dataset = Subset(dataset.test_dataset, test_subset_indices)

        # 【変更点1】 ラベルの再マッピングを無効化
        # 全クラス対応のHeadを使うため、ラベル(0~99)をそのまま使います。
        # if remap_labels: ... (無効化)

        if return_classifier:
            # 【変更点2】 共通の全クラスHeadを作成・ロードする
            # Task Vector等でマージする場合、初期値が全タスクで統一されている必要があります。
            
            head_init_path = f'checkpoints/{args.model}/{args.dataset}_full_head_init.pt'
            os.makedirs(os.path.dirname(head_init_path), exist_ok=True)

            if os.path.exists(head_init_path):
                print(f"Loading common initialization head from {head_init_path}")
                classification_head = ClassificationHead.load(head_init_path)
                classification_head = classification_head.to(args.device)
            else:
                print(f"Creating and saving common initialization head to {head_init_path}")
                # dataset.classnames は全クラス名を含んでいる前提
                classification_head = build_classification_head(
                    text_encoder.model, 
                    args.dataset, 
                    args.data_location, 
                    args.device, 
                    classnames=dataset.classnames
                )
                classification_head.save(head_init_path)

    else:
        raise NotImplementedError()
    
    # データローダーの作成
    dataset.train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset, batch_size=dataset.train_loader.batch_size,
        shuffle=True, num_workers=dataset.train_loader.num_workers
    )        
    dataset.test_loader = torch.utils.data.DataLoader(
        dataset.test_dataset, batch_size=dataset.test_loader.batch_size,
        shuffle=False, num_workers=dataset.test_loader.num_workers
    )

    # 【変更点3】 Datasetオブジェクトにアクティブクラス情報を付与
    dataset.active_classes = active_classes

    return (dataset, classification_head) if return_classifier else dataset