import gc
import time

import numpy as np

from .models import ResultStats, TaskResult


# 共有データとして大きな配列を作成
def create_shared_data(size: int) -> np.ndarray:
    """テスト用の大きな共有配列データを作成"""
    print(f"{size}x{size}の共有データ配列を作成中...")
    rng = np.random.default_rng()
    return rng.random((size, size))


# プロセス/スレッドで処理する関数 - 配列の一部に対する操作
def process_subarray(
    data: np.ndarray, start_idx: int, end_idx: int, operation_type: str
) -> TaskResult:
    """
    配列の一部（start_idx:end_idx）に対して演算を実行する

    引数:
        data: 処理する配列データ（全体）
        start_idx: 処理を開始する行インデックス
        end_idx: 処理を終了する行インデックス
        operation_type: 実行する演算のタイプ

    戻り値:
        TaskResult: 処理結果と処理時間
    """
    # 処理開始
    start_time: float = time.time()

    # サブ配列を取得（コピーではなく、ビュー）
    sub_data: np.ndarray = data[start_idx:end_idx, :]

    # 要求された演算を実行
    if operation_type == "matrix_product":
        # 行列積（GILを解放するNumPy演算）
        result: np.ndarray = np.dot(sub_data, sub_data.T)
    elif operation_type == "fft":
        # FFT（GILを解放するNumPy演算）
        result: np.ndarray = np.fft.fft2(sub_data)
    elif operation_type == "element_wise":
        # 要素ごとの演算（GILを解放するNumPy演算）
        result: np.ndarray = np.exp(np.sin(sub_data) + np.cos(sub_data))
    elif operation_type == "python_loops":
        # Pythonループを使った演算（GILの影響を強く受ける）
        result: np.ndarray = np.zeros_like(sub_data)
        for i in range(sub_data.shape[0]):
            for j in range(sub_data.shape[1]):
                result[i, j] = np.sin(sub_data[i, j]) + np.cos(sub_data[i, j])
    else:
        raise ValueError(f"未知の演算タイプ: {operation_type}")

    elapsed_time: float = time.time() - start_time

    # 処理結果のサマリー（配列全体を返すのではなく、特徴量を返す）
    result_summary = ResultStats(
        sum=float(result.sum()),
        min=float(result.min()),
        max=float(result.max()),
        mean=float(result.mean()),
        std=float(result.std()),
    )

    # ガベージコレクションを実行してメモリを開放
    gc.collect()

    return TaskResult(
        start_idx=start_idx, end_idx=end_idx, time=elapsed_time, result=result_summary
    )
