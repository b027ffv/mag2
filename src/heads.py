import os
import torch
from tqdm import tqdm

import open_clip

from src.datasets.templates import get_templates
from src.datasets.registry import get_dataset
from src.modeling import ClassificationHead, ImageEncoder


def build_classification_head(model, dataset_name, data_location, device, classnames):
    # クラス名リストの取得
    if not classnames:
        # datasetからクラス数を取得する処理が必要
        # ...
        dataset = get_dataset(dataset_name, None, location=data_location)
        classnames = dataset.classnames
    
    num_classes = len(classnames)
    embed_dim = model.num_features # timmモデルの特徴量次元数 (ViT-Bなら768)

    # ランダムな重みで初期化 (またはゼロ初期化)
    # MagMaxのClassificationHeadは重み行列を受け取る設計のため、それに合わせる
    random_weights = torch.randn(num_classes, embed_dim).to(device)
    
    # 正規化はしない方が良い場合が多いですが、MagMaxの構造に合わせるならTrue
    classification_head = ClassificationHead(normalize=True, weights=random_weights)

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
    if isinstance(image_encoder, ImageEncoder) and image_encoder.has_lang():
        print('Using passed model to create classifier!')
        model = image_encoder.model
    else:
        model = ImageEncoder(args, keep_lang=True).model

    classification_head = build_classification_head(model, dataset_name, args.data_location, args.device, classnames)
    
    return classification_head
