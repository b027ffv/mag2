import numpy as np
from collections import deque

class LossLandscapeDetector:
    def __init__(self, window_length=30, slope_threshold=3e-3, peak_factor=3.0, min_task_duration=50):
        self.window_length = window_length
        self.slope_threshold = slope_threshold # 閾値を少し緩める (1e-3 -> 3e-3 など)
        self.peak_factor = peak_factor
        self.min_task_duration = min_task_duration # 各タスクの最低保証ステップ数
        
        self.loss_window = deque(maxlen=window_length)
        self.step_window = deque(maxlen=window_length)
        
        self.current_task_min_mean = float('inf')
        self.current_task_variance = 0.0
        
        self.in_peak_region = True 
        self.total_steps = 0
        
        # 現在のタスクが始まってからのステップ数
        self.steps_in_current_task = 0

    def update(self, current_loss):
        self.loss_window.append(current_loss)
        self.step_window.append(self.total_steps)
        self.total_steps += 1
        self.steps_in_current_task += 1
        
        # Windowが埋まるまでは判定しない
        if len(self.loss_window) < self.window_length:
            return False, False

        window_mean = np.mean(self.loss_window)
        window_std = np.std(self.loss_window)
        
        is_plateau = False
        is_peak_start = False

        # --- 1. Peak (新タスク) 検知 ---
        # 判定基準: 現在のタスクの最小値からの乖離
        if not self.in_peak_region:
            dynamic_threshold = self.current_task_min_mean + (self.peak_factor * np.sqrt(self.current_task_variance))
            min_rise = 0.5 
            
            if window_mean > dynamic_threshold and window_mean > self.current_task_min_mean + min_rise:
                self.in_peak_region = True
                is_peak_start = True
                
                # 新タスク開始処理
                self.current_task_min_mean = float('inf')
                self.steps_in_current_task = 0  # タスク内ステップリセット
                self.loss_window.clear() # ウィンドウもクリアして前のタスクの影響を消す
                self.step_window.clear()
                return False, True # ここでリターン

        # --- 2. Plateau (収束) 検知 ---
        # 条件: 
        # A. ピーク領域にいる
        # B. タスク開始から最低期間が経過している (Step 39対策)
        if self.in_peak_region and self.steps_in_current_task > self.min_task_duration:
            
            # 最小値を記録
            if window_mean < self.current_task_min_mean:
                self.current_task_min_mean = window_mean
                self.current_task_variance = window_std ** 2

            # 【改良点】移動平均を使ってノイズを除去してから傾きを計算
            # window内のデータを平滑化
            data_array = np.array(self.loss_window)
            if len(data_array) >= 5:
                # 5点移動平均
                kernel = np.ones(5) / 5.0
                smoothed_data = np.convolve(data_array, kernel, mode='valid')
                # stepも合わせる
                smoothed_steps = np.array(self.step_window)[2:-2] 
            else:
                smoothed_data = data_array
                smoothed_steps = self.step_window

            # 平滑化したデータで傾き計算
            if len(smoothed_data) > 1:
                slope, _ = np.polyfit(smoothed_steps, smoothed_data, 1)
            else:
                slope = 1.0 # データ不足時は収束とみなさない

            # 判定ロジック
            # 1. 傾きの絶対値が閾値以下
            is_flat = abs(slope) < self.slope_threshold
            
            # 2. 現在の値が過去の最小値付近であること (上昇中の誤検知防止)
            #    分散が大きいときは許容範囲を広げる
            allowance = max(2.0 * np.sqrt(self.current_task_variance), 0.1)
            is_near_min = window_mean < self.current_task_min_mean + allowance
            
            if is_flat and is_near_min:
                self.in_peak_region = False
                is_plateau = True

        return is_plateau, is_peak_start

    @property
    def current_slope(self):
        """デバッグ表示用（生データの傾き）"""
        if len(self.loss_window) < self.window_length:
            return 0.0
        slope, _ = np.polyfit(self.step_window, self.loss_window, 1)
        return slope


"""
# src/online_utils.py として保存してください
import numpy as np
from collections import deque

class LossLandscapeDetector:
    """
    #Online-LoRAの損失曲面に基づくタスク境界検知ロジック
    #Ref: christina200/online-lora-official/.../Disjoint/engine.py
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
        #現在のバッチの損失を受け取り、境界（重要度更新タイミング）かどうかを判定する
        #Returns:
        #    bool: True if task boundary (plateau) is detected
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
        """