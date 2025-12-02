import torch
import os
from src.args import parse_arguments
from src.modeling import ImageEncoder

# 設定の読み込み (引数は適宜調整してください)
args = parse_arguments()
args.model = 'ViT-B-16' # 使用しているモデル名

# 保存先ディレクトリ
save_path = f'checkpoints/{args.model}/zeroshot.pt'

# 新しい(timm版の)ImageEncoderを作成
print(f"Creating new zeroshot model for {args.model}...")
image_encoder = ImageEncoder(args, keep_lang=True) # keep_langは無視されますがそのままでOK

# 保存
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

image_encoder.save(save_path)
print(f"Saved new zeroshot checkpoint to {save_path}")