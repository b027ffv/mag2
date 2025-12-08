import numpy as np
import torch
import wandb
import os
import tqdm
import copy

from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.eval import eval_task_aware, eval_task_agnostic
from src.args import parse_arguments
from src.modeling import ImageClassifier, ClassificationHead
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize
from src import utils 

# Config
args = parse_arguments()
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'

# --- 【変更】 ヘッドをマージせず、タスクベクトルのリストとしてロードする ---
def load_head_task_vectors(args, head_init_path, suffix=""):
    print(f"Loading common init head from {head_init_path}")
    if not os.path.exists(head_init_path):
        raise FileNotFoundError(f"Init head not found at {head_init_path}. Run finetune first.")
    
    # 初期ヘッドのロード (重み差分計算用)
    head_init = torch.load(head_init_path) # theta_init
    
    head_task_vectors = []
    # ここで timestamp をパスに含める
    base_dir = f'{args.save}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}-{args.timestamp}{suffix}'
    print(f"Loading heads from {base_dir} and creating TaskVectors...")

    for split_idx in range(args.n_splits):
        head_path = os.path.join(base_dir, f'head_{split_idx}.pt')
        
        if os.path.exists(head_path):
            try:
                task_head = ClassificationHead.load(head_path)
            except:
                task_head = torch.load(head_path)
        else:
            raise FileNotFoundError(f"Head checkpoint not found: {head_path}")
            
        # 差分計算 (Delta = theta_task_i - theta_init)
        # TaskVectorクラスは辞書形式のベクトルを受け取る
        vector_dict = {
            'weight': task_head.weight - head_init.weight,
            'bias': task_head.bias - head_init.bias
        }
        # リストに追加
        head_task_vectors.append(TaskVector(vector=vector_dict))

    return head_task_vectors

# --- 評価関数 (変更なし) ---
def eval_with_head(image_encoder, classification_head, dataset_name, args):
    model = ImageClassifier(image_encoder, classification_head)
    
    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)

    correct, n = 0., 0.
    model.eval()
    model.to(args.device)
    
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            y = data['labels'].to(args.device)

            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(args.device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

    metrics = {'top1': correct / n}
    return metrics


# --- 【修正】 マージ探索関数 (Headも同じメソッドでマージする) ---
def search_evaluate_merging(encoder_task_vectors, head_task_vectors, head_init_path, dataset, n_splits, split_strategy, n_coeffs=20):
    print(f"\nEVAL: {dataset}-{n_splits} ({split_strategy} incremental)")
    
    # マージ手法のリスト
    funcs_and_coeffs = [
        (merge_rnd_mix, [1.0]),     # Random Mix
        (merge_max_abs, [0.5]),     # Max Abs
        (sum, [1.0/n_splits]),      # Average (Sum * 1/N)
    ]

    for f, coeffs in funcs_and_coeffs:
        print(f"\nMerging with function: {f.__name__}")
        
        # 1. Encoderのマージ
        merged_encoder_tv = f(encoder_task_vectors)
        
        # 2. Headのマージ (同じ関数 f を適用！)
        merged_head_tv = f(head_task_vectors)
        
        results = {}
        for scaling_coef in coeffs:
            print(f"Scaling coeff: {scaling_coef}")
            
            # Encoderの適用
            image_encoder = merged_encoder_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            
            # Headの適用 (TaskVector.apply_to はパスを受け取ってロード＆適用してくれる)
            classification_head = merged_head_tv.apply_to(head_init_path, scaling_coef=scaling_coef)
            
            # 評価
            _r = eval_with_head(image_encoder, classification_head, dataset, args)['top1']
            
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
    
    # 1. EncoderのTIESマージ
    encoder_flat = torch.vstack([state_dict_to_vector(tv.vector) for tv in encoder_task_vectors])
    merged_encoder_flat = merge_methods(
        reset_type, encoder_flat, reset_thresh=reset_thresh, resolve_method=resolve, merge_func=merge
    )
    merged_encoder_vector = vector_to_state_dict(
        merged_encoder_flat, encoder_task_vectors[0].vector, remove_keys=[]
    )
    merged_encoder_tv = TaskVector(vector=merged_encoder_vector)

    # 2. HeadのTIESマージ (同じ設定で実行)
    head_flat = torch.vstack([state_dict_to_vector(tv.vector) for tv in head_task_vectors])
    merged_head_flat = merge_methods(
        reset_type, head_flat, reset_thresh=reset_thresh, resolve_method=resolve, merge_func=merge
    )
    merged_head_vector = vector_to_state_dict(
        merged_head_flat, head_task_vectors[0].vector, remove_keys=[]
    )
    merged_head_tv = TaskVector(vector=merged_head_vector)

    # 適用と評価
    results = {}
    for scaling_coef in [0.55]: # TIES推奨値
        print(f"Scaling coeff: {scaling_coef}")
        
        image_encoder = merged_encoder_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
        classification_head = merged_head_tv.apply_to(head_init_path, scaling_coef=scaling_coef)
        
        _r = eval_with_head(image_encoder, classification_head, dataset, args)['top1']
        
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

    name = f"merging-{args.dataset}-{args.n_splits}-{method}"

    wandb.init(
        project="magmax",
        group="merging-CIL",
        mode='online',
        name=name,
        config=args,
        tags=["merging", "CIL", f"{args.dataset}", f"{method}"],
    )
    
    # --- 変更1: Encoderの読み込みパスに timestamp を追加 ---
    encoder_task_vectors = [
        TaskVector(pretrained_checkpoint, f'{args.save}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}-{args.timestamp}{suffix}/finetuned_{_idx}.pt')
        for _idx in range(args.n_splits)
    ]
    # --------------------------------------------------
    
    # 2. Headのタスクベクトルをロード (リストとして取得)
    head_init_path = f'checkpoints/{args.model}/{args.dataset}_full_head_init.pt'
    head_task_vectors = load_head_task_vectors(args, head_init_path, suffix=suffix)
    
    print(f"Loaded {len(encoder_task_vectors)} encoder vectors and {len(head_task_vectors)} head vectors.")

    # 3. マージと評価を実行
    # ここでEncoderとHeadをペアにして、同じメソッドでマージ・評価します
    search_evaluate_merging(
        encoder_task_vectors, 
        head_task_vectors, 
        head_init_path, 
        args.dataset, 
        args.n_splits, 
        args.split_strategy
    )
"""import numpy as np
import torch
import wandb

from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.eval import eval_single_dataset, eval_task_aware, eval_task_agnostic
from src.args import parse_arguments


# Config
args = parse_arguments()
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'


def evaluate_individial_fts(task_vectors, args, task_agnostic=True):
    _eval_f = eval_task_agnostic if task_agnostic else eval_task_aware
    _eval_name = "task-agnostic" if task_agnostic else "task-aware"
    
    print('#' * 100 + f"\nPerforming {_eval_name} evaluation of individual finetunings.")

    results = []
    for idx in range(args.n_splits):
        print(f"\nEVAL: {args.dataset}-{args.n_splits} ({args.split_strategy} incremental) - split idx: {idx}")

        # Create the task vectors
        image_encoder = task_vectors[idx].apply_to(pretrained_checkpoint, scaling_coef=1.0)

        # Evaluate
        res = _eval_f(image_encoder, args)
        results.append(res)
        print(f"{_eval_name} eval on {args.dataset} after task {idx}. Accuracies:\n{res}")

    print(f"{_eval_name} evaluation of individual finetunings: final results:\n{results}\n" + '#' * 100 + '\n')


def evaluate_merged_fts(task_vectors, args, merging_f, scaling_coef, task_agnostic=True, only_final=False):
    _eval_f = eval_task_agnostic if task_agnostic else eval_task_aware
    _eval_name = "task-agnostic" if task_agnostic else "task-aware"
    
    print('#' * 100 + f"\nPerforming {_eval_name} evaluation of merged finetunings.")

    results = []
    for idx in range(args.n_splits):
        if only_final and idx != args.n_splits - 1:
            continue
        
        print(f"\nEVAL: {args.dataset}-{args.n_splits} ({args.split_strategy} incremental) - split idx: {idx}")

        _tvs = task_vectors[:idx+1]
        merged_tv = merging_f(_tvs)
        image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
        
        # Evaluate
        res = _eval_f(image_encoder, args)
        results.append(res)
        print(f"{_eval_name} eval on {args.dataset} after task {idx}. Accuracies:\n{res}")
    
    print(f"{_eval_name} evaluation of merged finetunings: final results:\n{results}\n" + '#' * 100 + '\n')


def search_evaluate_merging(task_vectors, dataset, n_splits, split_strategy, n_coeffs=20):
    print(f"\nEVAL: {dataset}-{n_splits} ({split_strategy} incremental)")
    
    funcs_and_coeffs = [
        # (merge_rnd_mix, np.linspace(0.5, 1.5, num=n_coeffs+1)[1:]),
        # (merge_max_abs, np.linspace(0.0, 1.0, num=n_coeffs+1)[1:]),
        # (sum, np.linspace(0.0, 2.0/n_splits, num=n_coeffs+1)[1:]),
        (merge_rnd_mix, [1.0]),
        (merge_max_abs, [0.5]),
        (sum, [1.0/n_splits]),
    ]

    for f, coeffs in funcs_and_coeffs:
        print(f"\nMerging with function: {f.__name__}")
        merged_tv = f(task_vectors)
        
        # Apply the resulting task vector
        results = {}
        for scaling_coef in coeffs:
            print(f"Scaling coeff: {scaling_coef}")
            image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            # Evaluate
            _r = eval_single_dataset(image_encoder, dataset, args)['top1']
            wandb.log({
                f"merging/{f.__name__}": _r * 100.0,
                "helpers/merging/alpha": scaling_coef,
            })
            results[scaling_coef] = _r

        print(f"Results with function {f.__name__}:\n{results}")
        
    # TIES merging
    reset_type = 'topk'
    reset_thresh = 20
    resolve = 'mass'
    merge = 'dis-mean'
    tv_flat_checks = torch.vstack([state_dict_to_vector(tv.vector) for tv in task_vectors])
    
    print(f"\nMerging with TIES merging: pruning {reset_type}-{reset_thresh}, resolve sign by {resolve}, merge by {merge}")
    
    merged_flat_tv = merge_methods(
        reset_type,
        tv_flat_checks,
        reset_thresh=reset_thresh,
        resolve_method=resolve,
        merge_func=merge,
    )
    merged_tv = vector_to_state_dict(
        merged_flat_tv, task_vectors[0].vector, remove_keys=[]
    )
    merged_tv = TaskVector(vector=merged_tv)

    # Apply the resulting task vector
    results = {}
    # for scaling_coef in np.linspace(0.5, 1.5, num=n_coeffs+1)[1:]:
    for scaling_coef in [0.55]:
        print(f"Scaling coeff: {scaling_coef}")
        image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
        # Evaluate
        _r = eval_single_dataset(image_encoder, dataset, args)['top1']
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

    name = f"merging-{args.dataset}-{args.n_splits}-{method}"

    wandb.init(
        project="magmax",
        group="merging-CIL",
        #entity=args.wandb_entity_name,
        mode='online',
        name=name,
        config=args,
        tags=["merging", "CIL", f"{args.dataset}", f"{method}"],
    )
    
    # preload task vectors
    task_vectors = [
        TaskVector(pretrained_checkpoint, f'{args.save}/{args.dataset}-{args.n_splits}/ft-epochs-{args.epochs}-seed:{args.seed}{suffix}/finetuned_{_idx}.pt')
        for _idx in range(args.n_splits)
    ]

    # evaluate_individial_fts(task_vectors, args, task_agnostic=False)
    # evaluate_individial_fts(task_vectors, args, task_agnostic=True)
    # evaluate_merged_fts(task_vectors, args, merge_max_abs, 0.5, task_agnostic=True)
    # evaluate_merged_fts(task_vectors, args, merge_max_abs, 0.5, task_agnostic=False)

    search_evaluate_merging(task_vectors, args.dataset, args.n_splits, args.split_strategy)
"""