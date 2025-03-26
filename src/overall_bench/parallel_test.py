import gc
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import psutil

from .data_processing import create_shared_data, process_subarray
from .models import TaskResult, TestResult


# 並列実行テスト
def run_parallel_test(
    data: np.ndarray,
    n_jobs: int,
    chunk_size: int,
    executor_class: type,
    operation_type: str,
) -> TestResult:
    """
    指定されたエグゼキュータを使用して並列処理のテストを実行

    引数:
        data: 処理する共有データ配列
        n_jobs: 並列ジョブ数
        chunk_size: 各ジョブが処理する行数
        executor_class: 使用するエグゼキュータクラス
        operation_type: 実行する演算のタイプ

    戻り値:
        TestResult: テスト結果
    """
    executor_name: str = executor_class.__name__
    total_size: int = data.shape[0]

    print(f"{executor_name}で{n_jobs}並列処理を実行中（演算: {operation_type}）...")

    # 測定前にガベージコレクションを実行して正確なメモリ使用量を計測
    gc.collect()

    # 開始時のメモリ使用量を記録
    start_memory: float = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    # 全体の処理開始時間
    start_time: float = time.time()

    # 各ジョブの開始・終了インデックスを計算
    jobs: list[tuple[int, int]] = []
    for i in range(n_jobs):
        start_idx: int = i * chunk_size
        end_idx: int = min(start_idx + chunk_size, total_size)
        jobs.append((start_idx, end_idx))

    results: list[TaskResult] = []

    # エグゼキュータを使用して並列処理を実行
    with executor_class(max_workers=n_jobs) as executor:
        # 測定前にガベージコレクションを実行
        gc.collect()

        # 実行中のメモリ使用量を記録
        executor_memory: float = psutil.Process(os.getpid()).memory_info().rss / (
            1024 * 1024
        )

        # 各ジョブを投入
        futures = []
        for start_idx, end_idx in jobs:
            future = executor.submit(
                process_subarray, data, start_idx, end_idx, operation_type
            )
            futures.append(future)

        # 結果を取得
        for future in futures:
            results.append(future.result())

    # 測定前にガベージコレクションを実行
    gc.collect()

    # 終了時のメモリ使用量
    end_memory: float = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    # 合計処理時間
    total_time: float = time.time() - start_time

    # 各ジョブの処理時間
    task_times: list[float] = [result.time for result in results]

    # ガベージコレクションを実行してメモリを開放
    gc.collect()

    # 結果をまとめる
    return TestResult(
        executor=executor_name,
        operation=operation_type,
        n_jobs=n_jobs,
        total_time=total_time,
        task_times=task_times,
        avg_task_time=float(np.mean(task_times)),
        max_task_time=float(np.max(task_times)),
        min_task_time=float(np.min(task_times)),
        start_memory=start_memory,
        executor_memory=executor_memory,
        end_memory=end_memory,
        memory_increase=end_memory - start_memory,
        results=results,
    )


# 比較テストを実行
def compare_executors(
    data_size=2000,
    n_jobs_list=[2, 4, 8],
    operations=["matrix_product", "fft", "element_wise", "python_loops"],
) -> list[TestResult]:
    """
    異なる条件下でProcessPoolExecutorとThreadPoolExecutorを比較

    引数:
        data_size: テストデータのサイズ
        n_jobs_list: テストする並列ジョブ数のリスト
        operations: テストする演算タイプのリスト

    戻り値:
        list[TestResult]: テスト結果のリスト
    """
    # ベンチマーク開始前にメモリをクリーンアップ
    gc.collect()

    # 共有データの作成
    data = create_shared_data(data_size)

    # 大きなデータ作成後にメモリ状況を安定させる
    gc.collect()

    all_results = []

    # 各演算タイプについてテスト
    for operation in operations:
        print(f"\n=== 演算: {operation} ===")

        for n_jobs in n_jobs_list:
            # 各テスト前にガベージコレクションを実行して測定の正確性を確保
            gc.collect()

            chunk_size: int = data_size // n_jobs

            # ProcessPoolExecutorのテスト
            process_result: TestResult = run_parallel_test(
                data, n_jobs, chunk_size, ProcessPoolExecutor, operation
            )
            all_results.append(process_result)

            # テスト間にガベージコレクションを実行して影響を排除
            gc.collect()

            # ThreadPoolExecutorのテスト
            thread_result: TestResult = run_parallel_test(
                data, n_jobs, chunk_size, ThreadPoolExecutor, operation
            )
            all_results.append(thread_result)

            # 結果の比較を表示
            process_time: float = process_result.total_time
            thread_time: float = thread_result.total_time
            speedup: float = (
                process_time / thread_time if thread_time > 0 else float("inf")
            )

            print(f"\n並列ジョブ数: {n_jobs}")
            print(f"ProcessPoolExecutor 合計時間: {process_time:.3f}秒")
            print(f"ThreadPoolExecutor 合計時間: {thread_time:.3f}秒")
            print(
                f"スレッド/プロセス 速度比: {speedup:.2f}x ({'ThreadPool有利' if speedup > 1 else 'ProcessPool有利'})"
            )

            # メモリ使用量
            print(
                f"ProcessPoolExecutor メモリ増加: {process_result.memory_increase:.1f} MB"
            )
            print(
                f"ThreadPoolExecutor メモリ増加: {thread_result.memory_increase:.1f} MB"
            )

            # 各テスト後にガベージコレクションを実行
            gc.collect()

    return all_results
