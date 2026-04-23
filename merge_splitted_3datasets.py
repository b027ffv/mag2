import os
import sys
import argparse
import re
import numpy as np
import torch
import wandb
import tqdm
from copy import deepcopy

from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.eval import eval_task_aware, eval_task_agnostic
from src.args import parse_arguments
from src.modeling import ImageClassifier, ClassificationHead
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize
from src import utils 

# --- 追加引数の処理 ---
custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument('--classes_per_task', type=int, default=10, help='Number of classes per task in multi-dataset mode')
custom_parser.add_argument('--num_cycles', type=int, default=3, help='Number of cycles across datasets')

custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

# Config
args = parse_arguments()
args.classes_per_task = custom_args.classes_per_task
args.num_cycles = custom_args.num_cycles

datasets_list = args.dataset.split(',')
is_multi_dataset = len(datasets_list) > 1
dataset_dir_name = args.dataset.replace(',', '_')

if is_multi_dataset:
    # フォルダ名構築用のためだけに使用
    args.n_splits = args.num_cycles * len(datasets_list)
    split_dir_name = f"{dataset_dir_name}-tasks:{args.n_splits}-cls_per_task:{args.classes_per_task}"
    head_init_name = f"multi_{dataset_dir_name}_full_head_init.pt"
else:
    split_dir_name = f"{args.dataset}-{args.n_splits}"
    head_init_name = f"{args.dataset}_full_head_init.pt"

pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'

# --- 【変更】 存在するインデックスのみをループしてロードする ---
def load_head_task_vectors(args, head_init_path, base_dir, found_indices):
    print(f"Loading common init head from {head_init_path}")
    if not os.path.exists(head_init_path):
        raise FileNotFoundError(f"Init head not found at {head_init_path}. Run finetune first.")
    
    # 初期ヘッドのロード (重み差分計算用)
    head_init = torch.load(head_init_path) # theta_init
    head_task_vectors = []
    
    print(f"Loading heads from {base_dir} and creating TaskVectors...")

    for _idx in found_indices:
        head_path = os.path.join(base_dir, f'head_{_idx}.pt')
        
        if os.path.exists(head_path):
            try:
                task_head = ClassificationHead.load(head_path)
            except:
                task_head = torch.load(head_path)
        else:
            raise FileNotFoundError(f"Head checkpoint not found: {head_path}")
            
        # 差分計算 (Delta = theta_task_i - theta_init)
        vector_dict = {
            'weight': (task_head.weight - head_init.weight).detach(),
            'bias': (task_head.bias - head_init.bias).detach()
        }
        head_task_vectors.append(TaskVector(vector=vector_dict))

    return head_task_vectors


def eval_with_head(image_encoder, classification_head, args):
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()
    model.to(args.device)
    
    correct, n = 0., 0.
    
    # 複数データセットのオフセットを計算
    class_offsets = {}
    total_classes = 0
    for ds_name in datasets_list:
        ds_obj = get_dataset(
            ds_name,
            model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size
        )
        class_offsets[ds_name] = total_classes
        total_classes += len(ds_obj.classnames)

    with torch.no_grad():
        # 全データセットのテストデータを評価して集計
        for ds_name in datasets_list:
            dataset = get_dataset(
                ds_name,
                model.val_preprocess,
                location=args.data_location,
                batch_size=args.batch_size
            )
            dataloader = get_dataloader(
                dataset, is_train=False, args=args, image_encoder=None)
            
            offset = class_offsets[ds_name]

            for data in tqdm.tqdm(dataloader, desc=f"Evaluating {ds_name}"):
                data = maybe_dictionarize(data)
                x = data['images'].to(args.device)
                y = data['labels'].to(args.device) + offset

                logits = utils.get_logits(x, model)
                pred = logits.argmax(dim=1, keepdim=True).to(args.device)
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

    metrics = {'top1': correct / n}
    return metrics


# --- マージ探索関数 ---
def search_evaluate_merging(encoder_task_vectors, head_task_vectors, head_init_path, num_tasks_merged, split_strategy, n_coeffs=20):
    print(f"\nEVAL: {split_dir_name} ({split_strategy} incremental) - Merging {num_tasks_merged} tasks")
    
    # マージ手法のリスト (sum は見つかった実際のタスク数で割る)
    funcs_and_coeffs = [
        (merge_rnd_mix, [1.0]),     # Random Mix
        (merge_max_abs, [0.5]),     # Max Abs
        (sum, [1.0/num_tasks_merged]), # Average (Sum * 1/N)
    ]

    for f, coeffs in funcs_and_coeffs:
        print(f"\nMerging with function: {f.__name__}")
        
        # 1. Encoderのマージ
        merged_encoder_tv = f(encoder_task_vectors)
        
        # 2. Headのマージ
        merged_head_tv = f(head_task_vectors)
        
        results = {}
        for scaling_coef in coeffs:
            print(f"Scaling coeff: {scaling_coef}")
            
            image_encoder = merged_encoder_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            classification_head = merged_head_tv.apply_to(head_init_path, scaling_coef=scaling_coef)
            
            # 評価 
            _r = eval_with_head(image_encoder, classification_head, args)['top1']
            
            wandb.log({
                f"merging/{f.__name__}": _r * 100.0,
                "helpers/merging/alpha": scaling_coef,
            })
            results[scaling_coef] = _r

        print(f"Results with function {f.__name__}:\n{results}")
        
    # --- TIES Merging ---
    print(f"\nMerging with TIES merging...")
    
    reset_type = 'topk'
    reset_thresh = 20
    resolve = 'mass'
    merge = 'dis-mean'
    
    encoder_flat = torch.vstack([state_dict_to_vector(tv.vector) for tv in encoder_task_vectors])
    merged_encoder_flat = merge_methods(
        reset_type, encoder_flat, reset_thresh=reset_thresh, resolve_method=resolve, merge_func=merge
    )
    merged_encoder_vector = vector_to_state_dict(
        merged_encoder_flat, encoder_task_vectors[0].vector, remove_keys=[]
    )
    merged_encoder_tv = TaskVector(vector=merged_encoder_vector)

    head_flat = torch.vstack([state_dict_to_vector(tv.vector) for tv in head_task_vectors])
    merged_head_flat = merge_methods(
        reset_type, head_flat, reset_thresh=reset_thresh, resolve_method=resolve, merge_func=merge
    )
    merged_head_vector = vector_to_state_dict(
        merged_head_flat, head_task_vectors[0].vector, remove_keys=[]
    )
    merged_head_tv = TaskVector(vector=merged_head_vector)

    results = {}
    for scaling_coef in [0.55]: # TIES推奨値
        print(f"Scaling coeff: {scaling_coef}")
        
        image_encoder = merged_encoder_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
        classification_head = merged_head_tv.apply_to(head_init_path, scaling_coef=scaling_coef)
        
        _r = eval_with_head(image_encoder, classification_head, args)['top1']
        
        wandb.log({
            f"merging/TIES": _r * 100.0,
            "helpers/merging/alpha": scaling_coef,
        })
        results[scaling_coef] = _r
            
    print(f"Results with function TIES:\n{results}")


if __name__ == '__main__':
    suffix = ""
    if args.lwf_lamb > 0.0:
        method = "lwf"
        args.save = f'checkpoints/{args.model}/lwf'
        suffix = f"-lamb:{args.lwf_lamb}"
    elif args.ewc_lamb > 0.0:
        method = "ewc"
        args.save = f'checkpoints/{args.model}/ewc'
        suffix = f"-lamb:{args.ewc_lamb}"
    elif args.sequential_finetuning:
        method = "seq-ft"
        args.save = f'checkpoints/{args.model}/sequential_finetuning/{args.split_strategy}_incremental'
    else:
        method = "ind-ft"
        args.save = f'checkpoints/{args.model}/{args.split_strategy}_incremental'

    name = f"merging-{dataset_dir_name}-{args.n_splits}-{method}"

    wandb.init(
        project="magmax",
        group="merging-CIL",
        mode='online',
        name=name,
        config=args,
        tags=["merging", "CIL", f"{dataset_dir_name}", f"{method}"],
    )
    
    # モデルのディレクトリパスの構築
    base_target_dir = f'{args.save}/{split_dir_name}/ft-epochs-{args.epochs}-seed:{args.seed}-{args.timestamp}{suffix}'
    print(f"Target Checkpoint Directory: {base_target_dir}")

    # ===== ★追加：ディレクトリ内に存在する全てのタスクベクトルを検索 =====
    found_indices = []
    pattern = re.compile(r'^finetuned_(\d+)\.pt$')
    
    if os.path.exists(base_target_dir):
        for fname in os.listdir(base_target_dir):
            match = pattern.match(fname)
            if match:
                found_indices.append(int(match.group(1)))
    
    found_indices.sort() # インデックス順にソート (0, 1, 2...)
    actual_num_tasks = len(found_indices)
    
    if actual_num_tasks == 0:
        raise FileNotFoundError(f"No task vectors (finetuned_*.pt) found in {base_target_dir}!")
        
    print(f"Found {actual_num_tasks} task vectors with indices: {found_indices}")
    # ====================================================================

    # 見つかったインデックスの Encoder タスクベクトルをロード
    encoder_task_vectors = [
        TaskVector(pretrained_checkpoint, os.path.join(base_target_dir, f'finetuned_{_idx}.pt'))
        for _idx in found_indices
    ]
    
    # 見つかったインデックスの Head タスクベクトルをロード
    head_init_path = f'checkpoints/{args.model}/{head_init_name}'
    head_task_vectors = load_head_task_vectors(args, head_init_path, base_dir=base_target_dir, found_indices=found_indices)
    
    print(f"Successfully loaded {len(encoder_task_vectors)} encoder vectors and {len(head_task_vectors)} head vectors.")

    # マージと評価を実行
    search_evaluate_merging(
        encoder_task_vectors, 
        head_task_vectors, 
        head_init_path, 
        actual_num_tasks, # 見つかったタスク数を渡す 
        args.split_strategy
    )