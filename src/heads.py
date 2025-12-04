import os
import torch
from tqdm import tqdm

import open_clip

from src.datasets.templates import get_templates
from src.datasets.registry import get_dataset
from src.modeling import ClassificationHead, ImageEncoder


def build_subset_classification_head(model, dataset_name, classes, data_location, device):
    """
    タスクに含まれるクラス数分の固定ランダムヘッドを作成します。
    """
    num_classes = len(classes)
    
    # モデルの特徴量次元数を取得
    if hasattr(model, 'num_features'):
        embed_dim = model.num_features
    else:
        # ImageEncoder経由などでラップされている場合
        embed_dim = 768  # ViT-B default, あるいは model.model.num_features等で取得

    print(f'Building SUBSET classification head (Orthogonal Init) for {num_classes} classes.')
    
    # 【重要】直交行列で初期化
    # 単純な randn よりも、各クラスのベクトルが直交（無相関）に近くなり、学習しやすくなります
    weights = torch.empty(num_classes, embed_dim, device=device)
    torch.nn.init.orthogonal_(weights)
    
    # 重みの正規化（ノルムを1にする）
    weights = weights / weights.norm(dim=-1, keepdim=True)
    
    # ClassificationHeadを作成 (normalize=Trueでコサイン類似度分類)
    classification_head = ClassificationHead(normalize=True, weights=weights)

    return classification_head


def build_subset_classification_head(model, dataset_name, classes, data_location, device):
    # テキストエンコーダーは使用せず、クラス数に合わせてランダムに初期化します
    
    # classesはタスクに含まれるクラスIDのリストです
    num_classes = len(classes)
    
    # timmモデルの特徴量次元数を取得 (ViT-Bなら768)
    # モデルによっては model.embed_dim などの場合もありますが、timmは通常 num_features です
    embed_dim = model.num_features 

    print(f'Building SUBSET classification head (Random Init) for {num_classes} classes.')
    
    # ランダムな重みで初期化 [出力クラス数 x 入力次元数]
    random_weights = torch.randn(num_classes, embed_dim).to(device)
    
    # ClassificationHeadを作成
    # normalize=Trueにすると、入力特徴量と重みのコサイン類似度ベースの分類になります
    # 通常の線形層に近づけたい場合は normalize=False にしてください
    classification_head = ClassificationHead(normalize=True, weights=random_weights)

    return classification_head


def get_classification_head(args, dataset_name, image_encoder=None, classnames=None):
    # 推論時などに呼ばれる関数ですが、MagMax(Fixed Head)では
    # merge_splitted.py で保存されたヘッドをロードして結合するため、
    # ここでは便宜的にランダムなヘッドを返すか、エラーにならないようにします。
    
    if image_encoder is None:
        model = ImageEncoder(args)
    else:
        model = image_encoder
    
    # データセットからクラス数を取得
    if not classnames:
        dataset = get_dataset(dataset_name, None, location=args.data_location)
        classnames = dataset.classnames
        
    return build_subset_classification_head(
        model, dataset_name, list(range(len(classnames))), args.data_location, args.device
    )
