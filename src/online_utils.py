# src/online_utils.py として保存してください
import numpy as np
from collections import deque

class LossLandscapeDetector:
    """
    Online-LoRAの損失曲面に基づくタスク境界検知ロジック
    Ref: christina200/online-lora-official/.../Disjoint/engine.py
    """
    def __init__(self, window_length, mean_threshold, var_threshold):
        self.window_length = window_length
        self.mean_threshold = mean_threshold
        self.var_threshold = var_threshold
        
        self.loss_window = deque(maxlen=window_length)
        self.last_loss_window_mean = 0.0
        self.last_loss_window_variance = 0.0
        
        # 新しいタスク（ピーク）が検知されたかどうかのフラグ
        # 初期状態はTrueにしておき、最初の減少を監視する
        self.new_peak_detected = True 

    def update(self, current_loss):
        """
        現在のバッチの損失を受け取り、境界（重要度更新タイミング）かどうかを判定する
        Returns:
            bool: True if task boundary (plateau) is detected
        """
        # Windowの更新
        self.loss_window.append(current_loss)
        is_plateau = False
        is_peak_start = False
        
        # Windowが埋まるまでは判定しない
        if len(self.loss_window) < self.window_length:
            return False , False

        # 平均と分散の計算
        loss_window_mean = np.mean(self.loss_window)
        loss_window_variance = np.var(self.loss_window)
        
        # ピーク検知ロジック (Online-LoRA engine.py より)
        # 前回の安定状態よりも損失が大きく上がった場合、新しいタスク(ピーク)とみなす
        if not self.new_peak_detected:
            if loss_window_mean > self.last_loss_window_mean + np.sqrt(self.last_loss_window_variance):
                self.new_peak_detected = True
                is_peak_start = True # ピーク開始フラグを立てる
                # print("DEBUG: New peak (loss increase) detected.")

        # 境界検知ロジック
        # 1. 新しいピークの中にいる (new_peak_detected is True)
        # 2. 平均損失が閾値を下回った (十分学習が進んだ)
        # 3. 分散が閾値を下回った (学習が安定した)
        if self.new_peak_detected and \
           loss_window_mean < self.mean_threshold and \
           loss_window_variance < self.var_threshold:
            
            # 状態の更新
            self.last_loss_window_mean = loss_window_mean
            self.last_loss_window_variance = loss_window_variance
            self.new_peak_detected = False # フラグをリセット
            
            is_plateau = True

        return is_plateau, is_peak_start