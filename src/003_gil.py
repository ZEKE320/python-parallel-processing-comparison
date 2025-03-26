# %%
import gc
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime  # 日時情報の取得用に追加
from pathlib import Path  # pathlibをインポート

import japanize_matplotlib  # 日本語フォント設定
import matplotlib.pyplot as plt
import numpy as np
import psutil

# ベンチマーク識別子（定数）
BENCHMARK_NO = "003"
BENCHMARK_NAME = "gil"

# 日本語表示用のマッピング
TASK_TYPE_NAMES: dict[str, str] = {
    "python_intensive_task": "Python集中型タスク",
    "numpy_optimized_task": "NumPy最適化タスク",
    "hybrid_task": "ハイブリッドタスク",
}

# スレッド数とプロセス数の上限
MAX_WORKERS: int = min(multiprocessing.cpu_count(), 8)


# GILの影響を受ける純粋なPythonコード
def python_intensive_task(
    data: np.ndarray, start_idx: int, end_idx: int, iterations: int = 5
) -> dict[str, object]:
    """
    GILの影響を受ける純粋なPythonコードによる処理

    引数:
        data: 処理するNumPy配列
        start_idx: 処理を開始する行インデックス
        end_idx: 処理を終了する行インデックス
        iterations: 反復回数

    戻り値:
        処理時間と結果の統計量
    """
    start_time: float = time.time()

    # データのスライスを取得（ビュー）
    sub_data: np.ndarray = data[start_idx:end_idx, :]
    rows: int = sub_data.shape[0]
    cols: int = sub_data.shape[1]

    # 結果格納用の配列を初期化
    result: np.ndarray = np.zeros_like(sub_data)

    # GILを保持するPythonレベルのループ
    for _ in range(iterations):
        for i in range(rows):
            for j in range(cols):
                # Python演算（GILを保持する）
                value = sub_data[i, j]
                result[i, j] += np.sin(value) + np.cos(value) + np.sqrt(abs(value))

    elapsed_time: float = time.time() - start_time

    # 結果の統計量を返す
    result_stats: dict[str, float] = {
        "sum": float(np.sum(result)),
        "mean": float(np.mean(result)),
        "min": float(np.min(result)),
        "max": float(np.max(result)),
    }

    return {"time": elapsed_time, "stats": result_stats, "rows_processed": rows}


# GILを解放するNumPy最適化コード
def numpy_optimized_task(
    data: np.ndarray, start_idx: int, end_idx: int, iterations: int = 5
) -> dict[str, object]:
    """
    GILを解放するNumPyの最適化コードによる処理

    引数:
        data: 処理するNumPy配列
        start_idx: 処理を開始する行インデックス
        end_idx: 処理を終了する行インデックス
        iterations: 反復回数

    戻り値:
        処理時間と結果の統計量
    """
    start_time: float = time.time()

    # データのスライスを取得（ビュー）
    sub_data: np.ndarray = data[start_idx:end_idx, :]

    # 結果格納用の配列を初期化
    result: np.ndarray = np.zeros_like(sub_data)

    # NumPyのベクトル化された演算（GILを解放する）
    for _ in range(iterations):
        # すべての演算が一度にベクトル化される
        result += np.sin(sub_data) + np.cos(sub_data) + np.sqrt(np.abs(sub_data))

    elapsed_time: float = time.time() - start_time

    # 結果の統計量を返す
    result_stats: dict[str, float] = {
        "sum": float(np.sum(result)),
        "mean": float(np.mean(result)),
        "min": float(np.min(result)),
        "max": float(np.max(result)),
    }

    return {
        "time": elapsed_time,
        "stats": result_stats,
        "rows_processed": sub_data.shape[0],
    }


# ハイブリッド処理（Pythonループと最適化NumPy演算の混合）
def hybrid_task(
    data: np.ndarray,
    start_idx: int,
    end_idx: int,
    iterations: int = 5,
    numpy_ratio: float = 0.5,
) -> dict[str, object]:
    """
    PythonコードとNumPy最適化コードを混合したハイブリッド処理

    引数:
        data: 処理するNumPy配列
        start_idx: 処理を開始する行インデックス
        end_idx: 処理を終了する行インデックス
        iterations: 反復回数
        numpy_ratio: NumPy最適化コードの比率 (0.0～1.0)

    戻り値:
        処理時間と結果の統計量
    """
    start_time: float = time.time()

    # データのスライスを取得（ビュー）
    sub_data: np.ndarray = data[start_idx:end_idx, :]
    rows: int = sub_data.shape[0]
    cols: int = sub_data.shape[1]

    # 結果格納用の配列を初期化
    result: np.ndarray = np.zeros_like(sub_data)

    # 各反復での処理
    for _ in range(iterations):
        # NumPy最適化コードの部分（GILを解放）
        numpy_rows: int = int(rows * numpy_ratio)
        if numpy_rows > 0:
            numpy_slice: np.ndarray = sub_data[:numpy_rows, :]
            result[:numpy_rows, :] += (
                np.sin(numpy_slice) + np.cos(numpy_slice) + np.sqrt(np.abs(numpy_slice))
            )

        # Pythonループの部分（GILを保持）
        python_rows: int = rows - numpy_rows
        if python_rows > 0:
            for i in range(numpy_rows, rows):
                for j in range(cols):
                    value = sub_data[i, j]
                    result[i, j] += np.sin(value) + np.cos(value) + np.sqrt(abs(value))

    elapsed_time: float = time.time() - start_time

    # 結果の統計量を返す
    result_stats: dict[str, float] = {
        "sum": float(np.sum(result)),
        "mean": float(np.mean(result)),
        "min": float(np.min(result)),
        "max": float(np.max(result)),
    }

    return {
        "time": elapsed_time,
        "stats": result_stats,
        "rows_processed": rows,
        "numpy_ratio": numpy_ratio,
    }


def run_parallel_test(
    data: np.ndarray,
    task_func: callable,
    executor_class: type,
    n_workers: int,
    iterations: int = 5,
    extra_args: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    並列処理のテストを実行

    引数:
        data: 処理するNumPy配列
        task_func: 実行するタスク関数
        executor_class: 使用するエグゼキュータクラス
        n_workers: ワーカー数
        iterations: 各タスクでの反復回数
        extra_args: タスク関数に渡す追加引数の辞書

    戻り値:
        テスト結果の辞書
    """
    executor_name: str = executor_class.__name__
    total_rows: int = data.shape[0]

    # ワーカーごとの作業範囲を計算
    chunk_size: int = total_rows // n_workers
    tasks: list[tuple[int, int]] = []
    for i in range(n_workers):
        start_idx: int = i * chunk_size
        end_idx: int = start_idx + chunk_size if i < n_workers - 1 else total_rows
        tasks.append((start_idx, end_idx))

    # テスト実行
    start_time: float = time.time()

    results: list[dict[str, object]] = []
    with executor_class(max_workers=n_workers) as executor:
        # 各ワーカーにタスクを投入
        futures = []
        for start_idx, end_idx in tasks:
            if extra_args:
                future = executor.submit(
                    task_func, data, start_idx, end_idx, iterations, **extra_args
                )
            else:
                future = executor.submit(
                    task_func, data, start_idx, end_idx, iterations
                )
            futures.append(future)

        # 結果を収集
        for future in futures:
            results.append(future.result())

    total_time: float = time.time() - start_time

    # 各タスクの処理時間を集計
    task_times: list[float] = [result["time"] for result in results]

    return {
        "executor": executor_name,
        "task_type": task_func.__name__,
        "n_workers": n_workers,
        "iterations": iterations,
        "total_time": total_time,
        "task_times": task_times,
        "avg_task_time": float(np.mean(task_times)),
        "max_task_time": float(np.max(task_times)),
        "min_task_time": float(np.min(task_times)),
        "results": results,
        "extra_args": extra_args,
    }


def compare_gil_impact(
    data_size: int = 1000,
    worker_counts: list[int] | None = None,
    iterations: int = 5,
    numpy_ratios: list[float] | None = None,
) -> list[dict[str, object]]:
    """
    GILの影響を検証するための比較テストを実行

    引数:
        data_size: テストデータの行数・列数
        worker_counts: テストするワーカー数のリスト
        iterations: 各タスクでの反復回数
        numpy_ratios: ハイブリッドタスクでテストするNumPy比率のリスト

    戻り値:
        テスト結果のリスト
    """
    if worker_counts is None:
        worker_counts = [1, 2, 4, 8]

    if numpy_ratios is None:
        numpy_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    # テストデータの作成
    print(f"{data_size}x{data_size}のテストデータを生成中...")
    data: np.ndarray = np.random.random((data_size, data_size))

    # 結果格納用
    all_results: list[dict[str, object]] = []

    # 1. 純粋なPythonコード - ワーカー数を変えてテスト
    print("\n=== 純粋なPythonコード（GILの影響を受ける）===")
    for n_workers in worker_counts:
        if n_workers > MAX_WORKERS:
            continue

        print(f"\nワーカー数: {n_workers}")

        # ProcessPoolExecutorでテスト
        process_result: dict[str, object] = run_parallel_test(
            data, python_intensive_task, ProcessPoolExecutor, n_workers, iterations
        )
        all_results.append(process_result)

        # ThreadPoolExecutorでテスト
        thread_result: dict[str, object] = run_parallel_test(
            data, python_intensive_task, ThreadPoolExecutor, n_workers, iterations
        )
        all_results.append(thread_result)

        # 結果の比較
        process_time: float = process_result["total_time"]
        thread_time: float = thread_result["total_time"]
        speedup: float = (
            thread_time / process_time if process_time > 0 else float("inf")
        )

        print(f"ProcessPoolExecutor 時間: {process_time:.3f}秒")
        print(f"ThreadPoolExecutor 時間: {thread_time:.3f}秒")
        print(f"ProcessPool/ThreadPool 速度比: {speedup:.2f}x")

    # 2. NumPyの最適化コード - ワーカー数を変えてテスト
    print("\n=== NumPy最適化コード（GILを解放する）===")
    for n_workers in worker_counts:
        if n_workers > MAX_WORKERS:
            continue

        print(f"\nワーカー数: {n_workers}")

        # ProcessPoolExecutorでテスト
        process_result: dict[str, object] = run_parallel_test(
            data, numpy_optimized_task, ProcessPoolExecutor, n_workers, iterations
        )
        all_results.append(process_result)

        # ThreadPoolExecutorでテスト
        thread_result: dict[str, object] = run_parallel_test(
            data, numpy_optimized_task, ThreadPoolExecutor, n_workers, iterations
        )
        all_results.append(thread_result)

        # 結果の比較
        process_time: float = process_result["total_time"]
        thread_time: float = thread_result["total_time"]
        speedup: float = process_time / thread_time if thread_time > 0 else float("inf")

        print(f"ProcessPoolExecutor 時間: {process_time:.3f}秒")
        print(f"ThreadPoolExecutor 時間: {thread_time:.3f}秒")
        print(
            f"ProcessPool/ThreadPool 速度比: {speedup:.2f}x ({'ThreadPool有利' if speedup > 1 else 'ProcessPool有利'})"
        )

    # 3. ハイブリッドコード - NumPy比率を変えてテスト
    print("\n=== ハイブリッドコード（NumPy比率を変化させる）===")
    n_workers: int = 4  # 固定ワーカー数

    for numpy_ratio in numpy_ratios:
        print(f"\nNumPy比率: {numpy_ratio:.2f}")

        extra_args: dict[str, float] = {"numpy_ratio": numpy_ratio}

        # ProcessPoolExecutorでテスト
        process_result: dict[str, object] = run_parallel_test(
            data, hybrid_task, ProcessPoolExecutor, n_workers, iterations, extra_args
        )
        all_results.append(process_result)

        # ThreadPoolExecutorでテスト
        thread_result: dict[str, object] = run_parallel_test(
            data, hybrid_task, ThreadPoolExecutor, n_workers, iterations, extra_args
        )
        all_results.append(thread_result)

        # 結果の比較
        process_time: float = process_result["total_time"]
        thread_time: float = thread_result["total_time"]
        speedup: float = process_time / thread_time if thread_time > 0 else float("inf")

        print(f"ProcessPoolExecutor 時間: {process_time:.3f}秒")
        print(f"ThreadPoolExecutor 時間: {thread_time:.3f}秒")
        print(
            f"ProcessPool/ThreadPool 速度比: {speedup:.2f}x ({'ThreadPool有利' if speedup > 1 else 'ProcessPool有利'})"
        )

    return all_results


def visualize_gil_impact(results: list[dict[str, object]]) -> None:
    """GIL影響テストの結果を可視化"""
    # 可視化前にメモリをクリーンアップ
    gc.collect()

    # 画像保存用ディレクトリを作成
    save_dir = Path("fig") / f"{BENCHMARK_NO}_{BENCHMARK_NAME}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 純粋なPythonコードとNumPy最適化コードの比較
    task_types: list[str] = ["python_intensive_task", "numpy_optimized_task"]
    worker_counts: list[int] = sorted(
        set(r["n_workers"] for r in results if r["task_type"] in task_types)
    )

    plt.figure(figsize=(12, 10))

    # 1.1 Pythonコード - ワーカー数による速度変化
    plt.subplot(2, 2, 1)

    process_times: list[float] = []
    thread_times: list[float] = []

    for n_workers in worker_counts:
        # プロセスプールの結果
        process_result = next(
            (
                r
                for r in results
                if r["task_type"] == "python_intensive_task"
                and r["executor"] == "ProcessPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            None,
        )
        if process_result:
            process_times.append(float(process_result["total_time"]))
        else:
            process_times.append(0)

        # スレッドプールの結果
        thread_result = next(
            (
                r
                for r in results
                if r["task_type"] == "python_intensive_task"
                and r["executor"] == "ThreadPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            None,
        )
        if thread_result:
            thread_times.append(float(thread_result["total_time"]))
        else:
            thread_times.append(0)

    plt.plot(worker_counts, process_times, "bo-", label="ProcessPool")
    plt.plot(worker_counts, thread_times, "ro-", label="ThreadPool")
    plt.xlabel("ワーカー数")
    plt.ylabel("実行時間 (秒)")
    plt.title("Pythonコード（GIL影響あり）")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 1.2 NumPyコード - ワーカー数による速度変化
    plt.subplot(2, 2, 2)

    process_times: list[float] = []
    thread_times: list[float] = []

    for n_workers in worker_counts:
        # プロセスプールの結果
        process_result = next(
            (
                r
                for r in results
                if r["task_type"] == "numpy_optimized_task"
                and r["executor"] == "ProcessPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            None,
        )
        if process_result:
            process_times.append(process_result["total_time"])
        else:
            process_times.append(0)

        # スレッドプールの結果
        thread_result = next(
            (
                r
                for r in results
                if r["task_type"] == "numpy_optimized_task"
                and r["executor"] == "ThreadPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            None,
        )
        if thread_result:
            thread_times.append(thread_result["total_time"])
        else:
            thread_times.append(0)

    plt.plot(worker_counts, process_times, "bo-", label="ProcessPool")
    plt.plot(worker_counts, thread_times, "ro-", label="ThreadPool")
    plt.xlabel("ワーカー数")
    plt.ylabel("実行時間 (秒)")
    plt.title("NumPyコード（GIL解放）")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 1.3 速度向上率の比較
    plt.subplot(2, 2, 3)

    python_speedups: list[float] = []
    numpy_speedups: list[float] = []

    for n_workers in worker_counts:
        # Pythonコードの速度比
        python_process = next(
            (
                r
                for r in results
                if r["task_type"] == "python_intensive_task"
                and r["executor"] == "ProcessPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            None,
        )
        python_thread = next(
            (
                r
                for r in results
                if r["task_type"] == "python_intensive_task"
                and r["executor"] == "ThreadPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            None,
        )

        if python_process and python_thread and python_thread["total_time"] > 0:
            python_speedup = python_process["total_time"] / python_thread["total_time"]
            python_speedups.append(python_speedup)
        else:
            python_speedups.append(0)

        # NumPyコードの速度比
        numpy_process = next(
            (
                r
                for r in results
                if r["task_type"] == "numpy_optimized_task"
                and r["executor"] == "ProcessPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            None,
        )
        numpy_thread = next(
            (
                r
                for r in results
                if r["task_type"] == "numpy_optimized_task"
                and r["executor"] == "ThreadPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            None,
        )

        if numpy_process and numpy_thread and numpy_thread["total_time"] > 0:
            numpy_speedup = numpy_process["total_time"] / numpy_thread["total_time"]
            numpy_speedups.append(numpy_speedup)
        else:
            numpy_speedups.append(0)

    width: float = 0.35
    x = np.arange(len(worker_counts))

    plt.bar(
        x - width / 2, python_speedups, width, label="Pythonコード", color="skyblue"
    )
    plt.bar(x + width / 2, numpy_speedups, width, label="NumPyコード", color="salmon")
    plt.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)

    plt.xlabel("ワーカー数")
    plt.ylabel("速度比 (Process/Thread)")
    plt.title("GILの影響による速度比 (>1: ThreadPool有利)")
    plt.xticks(x, worker_counts)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    # 1.4 ハイブリッドコードのNumPy比率による影響
    plt.subplot(2, 2, 4)

    # ハイブリッドタスクの結果を取得
    hybrid_results: list[dict[str, object]] = [
        r for r in results if r["task_type"] == "hybrid_task"
    ]
    numpy_ratios: list[float] = sorted(
        set(r["extra_args"]["numpy_ratio"] for r in hybrid_results if r["extra_args"])
    )

    process_times: list[float] = []
    thread_times: list[float] = []
    speedups: list[float] = []

    for ratio in numpy_ratios:
        # プロセスプールの結果
        process_result = next(
            (
                r
                for r in hybrid_results
                if r["executor"] == "ProcessPoolExecutor"
                and r["extra_args"]
                and r["extra_args"]["numpy_ratio"] == ratio
            ),
            None,
        )

        # スレッドプールの結果
        thread_result = next(
            (
                r
                for r in hybrid_results
                if r["executor"] == "ThreadPoolExecutor"
                and r["extra_args"]
                and r["extra_args"]["numpy_ratio"] == ratio
            ),
            None,
        )

        if process_result and thread_result:
            process_times.append(process_result["total_time"])
            thread_times.append(thread_result["total_time"])

            speedup = (
                process_result["total_time"] / thread_result["total_time"]
                if thread_result["total_time"] > 0
                else 0
            )
            speedups.append(speedup)

    plt.plot(numpy_ratios, process_times, "bo-", label="ProcessPool")
    plt.plot(numpy_ratios, thread_times, "ro-", label="ThreadPool")
    plt.plot(
        numpy_ratios,
        [t1 / t2 for t1, t2 in zip(process_times, thread_times)],
        "go-",
        label="速度比",
    )

    plt.xlabel("NumPy比率")
    plt.ylabel("実行時間 (秒)")
    plt.title("NumPy比率の影響")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "gil_impact_analysis.png", dpi=100)

    # 2. スケーラビリティの分析
    plt.figure(figsize=(12, 6))

    # 2.1 Pythonコードのスケーラビリティ
    plt.subplot(1, 2, 1)

    # 単一ワーカーの実行時間を基準にして相対的なスケーラビリティを計算
    python_process_base: float = next(
        (
            r["total_time"]
            for r in results
            if r["task_type"] == "python_intensive_task"
            and r["executor"] == "ProcessPoolExecutor"
            and r["n_workers"] == 1
        ),
        1,
    )
    python_thread_base: float = next(
        (
            r["total_time"]
            for r in results
            if r["task_type"] == "python_intensive_task"
            and r["executor"] == "ThreadPoolExecutor"
            and r["n_workers"] == 1
        ),
        1,
    )

    python_process_speedups: list[float] = []
    python_thread_speedups: list[float] = []

    for n_workers in worker_counts:
        # プロセスプールのスケーラビリティ
        process_time = next(
            (
                r["total_time"]
                for r in results
                if r["task_type"] == "python_intensive_task"
                and r["executor"] == "ProcessPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            0,
        )
        if process_time > 0:
            python_process_speedups.append(python_process_base / process_time)
        else:
            python_process_speedups.append(0)

        # スレッドプールのスケーラビリティ
        thread_time = next(
            (
                r["total_time"]
                for r in results
                if r["task_type"] == "python_intensive_task"
                and r["executor"] == "ThreadPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            0,
        )
        if thread_time > 0:
            python_thread_speedups.append(python_thread_base / thread_time)
        else:
            python_thread_speedups.append(0)

    plt.plot(worker_counts, python_process_speedups, "bo-", label="ProcessPool")
    plt.plot(worker_counts, python_thread_speedups, "ro-", label="ThreadPool")
    plt.plot(
        worker_counts, worker_counts, "k--", alpha=0.5, label="理想的なスケーラビリティ"
    )

    plt.xlabel("ワーカー数")
    plt.ylabel("スピードアップ係数")
    plt.title("Pythonコードのスケーラビリティ")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2.2 NumPyコードのスケーラビリティ
    plt.subplot(1, 2, 2)

    # 単一ワーカーの実行時間を基準にして相対的なスケーラビリティを計算
    numpy_process_base: float = next(
        (
            r["total_time"]
            for r in results
            if r["task_type"] == "numpy_optimized_task"
            and r["executor"] == "ProcessPoolExecutor"
            and r["n_workers"] == 1
        ),
        1,
    )
    numpy_thread_base: float = next(
        (
            r["total_time"]
            for r in results
            if r["task_type"] == "numpy_optimized_task"
            and r["executor"] == "ThreadPoolExecutor"
            and r["n_workers"] == 1
        ),
        1,
    )

    numpy_process_speedups: list[float] = []
    numpy_thread_speedups: list[float] = []

    for n_workers in worker_counts:
        # プロセスプールのスケーラビリティ
        process_time = next(
            (
                r["total_time"]
                for r in results
                if r["task_type"] == "numpy_optimized_task"
                and r["executor"] == "ProcessPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            0,
        )
        if process_time > 0:
            numpy_process_speedups.append(numpy_process_base / process_time)
        else:
            numpy_process_speedups.append(0)

        # スレッドプールのスケーラビリティ
        thread_time = next(
            (
                r["total_time"]
                for r in results
                if r["task_type"] == "numpy_optimized_task"
                and r["executor"] == "ThreadPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            0,
        )
        if thread_time > 0:
            numpy_thread_speedups.append(numpy_thread_base / thread_time)
        else:
            numpy_thread_speedups.append(0)

    plt.plot(worker_counts, numpy_process_speedups, "bo-", label="ProcessPool")
    plt.plot(worker_counts, numpy_thread_speedups, "ro-", label="ThreadPool")
    plt.plot(
        worker_counts, worker_counts, "k--", alpha=0.5, label="理想的なスケーラビリティ"
    )

    plt.xlabel("ワーカー数")
    plt.ylabel("スピードアップ係数")
    plt.title("NumPyコードのスケーラビリティ")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "gil_scalability_analysis.png", dpi=100)

    # 可視化完了後にメモリ開放
    plt.close("all")
    gc.collect()


def generate_markdown_report(
    results: list[dict[str, object]],
    data_size: int,
    worker_counts: list[int],
    iterations: int,
    numpy_ratios: list[float],
) -> str:
    """テスト結果をMarkdownレポートとして生成

    引数:
        results: テスト結果のリスト
        data_size: テストデータのサイズ
        worker_counts: テストしたワーカー数のリスト
        iterations: 各タスクでの反復回数
        numpy_ratios: テストしたNumPy比率のリスト

    戻り値:
        Markdownフォーマットのレポート
    """
    # システム情報
    cpu_count: int | None = os.cpu_count()
    mem_info = psutil.virtual_memory()

    # 現在の日時
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    # レポートのヘッダー
    report = [
        f"# GILの影響検証: ProcessPoolExecutor vs ThreadPoolExecutor",
        f"\n## 実行概要",
        f"- ベンチマーク ID: {BENCHMARK_NO}_{BENCHMARK_NAME}",
        f"- 実行日時: {now}",
        f"\n## システム情報",
        f"- CPU: {cpu_count}コア",
        f"- メモリ: {mem_info.total / (1024**3):.1f} GB",
        f"\n## テスト設定",
        f"- データサイズ: {data_size}x{data_size} の行列",
        f"- ワーカー数: {', '.join(map(str, worker_counts))}",
        f"- 反復回数: {iterations}",
        f"- NumPy比率 (ハイブリッドタスク): {', '.join([f'{ratio:.2f}' for ratio in numpy_ratios])}",
    ]

    # Python集中型タスクの結果
    report.append(f"\n## 1. Python集中型タスク (GIL影響あり)")

    py_task_results = [r for r in results if r["task_type"] == "python_intensive_task"]
    report.append(f"\n### ワーカー数による実行時間の比較")

    report.append(
        f"\n| ワーカー数 | ProcessPool (秒) | ThreadPool (秒) | 速度比 (Process/Thread) | 有利なエグゼキュータ |"
    )
    report.append(
        f"|----------:|----------------:|--------------:|----------------------:|:---------------------|"
    )

    for n_workers in worker_counts:
        process_result = next(
            (
                r
                for r in py_task_results
                if r["executor"] == "ProcessPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            None,
        )
        thread_result = next(
            (
                r
                for r in py_task_results
                if r["executor"] == "ThreadPoolExecutor" and r["n_workers"] == n_workers
            ),
            None,
        )

        if process_result and thread_result:
            process_time = process_result["total_time"]
            thread_time = thread_result["total_time"]
            speedup = thread_time / process_time if process_time > 0 else float("inf")

            report.append(
                f"| {n_workers} | {process_time:.3f} | {thread_time:.3f} | {1 / speedup:.2f}x | {'ThreadPool' if speedup < 1 else 'ProcessPool'} |"
            )

    # NumPy最適化タスクの結果
    report.append(f"\n## 2. NumPy最適化タスク (GIL解放)")

    numpy_task_results = [
        r for r in results if r["task_type"] == "numpy_optimized_task"
    ]
    report.append(f"\n### ワーカー数による実行時間の比較")

    report.append(
        f"\n| ワーカー数 | ProcessPool (秒) | ThreadPool (秒) | 速度比 (Process/Thread) | 有利なエグゼキュータ |"
    )
    report.append(
        f"|----------:|----------------:|--------------:|----------------------:|:---------------------|"
    )

    for n_workers in worker_counts:
        process_result = next(
            (
                r
                for r in numpy_task_results
                if r["executor"] == "ProcessPoolExecutor"
                and r["n_workers"] == n_workers
            ),
            None,
        )
        thread_result = next(
            (
                r
                for r in numpy_task_results
                if r["executor"] == "ThreadPoolExecutor" and r["n_workers"] == n_workers
            ),
            None,
        )

        if process_result and thread_result:
            process_time = process_result["total_time"]
            thread_time = thread_result["total_time"]
            speedup = process_time / thread_time if thread_time > 0 else float("inf")

            report.append(
                f"| {n_workers} | {process_time:.3f} | {thread_time:.3f} | {speedup:.2f}x | {'ThreadPool' if speedup > 1 else 'ProcessPool'} |"
            )

    # ハイブリッドタスクの結果
    report.append(f"\n## 3. ハイブリッドタスク (NumPy比率の影響)")

    hybrid_task_results = [r for r in results if r["task_type"] == "hybrid_task"]
    report.append(f"\n### NumPy比率による実行時間の比較 (ワーカー数: 4)")

    report.append(
        f"\n| NumPy比率 | ProcessPool (秒) | ThreadPool (秒) | 速度比 (Process/Thread) | 有利なエグゼキュータ |"
    )
    report.append(
        f"|----------:|----------------:|--------------:|----------------------:|:---------------------|"
    )

    for ratio in numpy_ratios:
        process_result = next(
            (
                r
                for r in hybrid_task_results
                if r["executor"] == "ProcessPoolExecutor"
                and r["extra_args"]
                and r["extra_args"].get("numpy_ratio") == ratio
            ),
            None,
        )

        thread_result = next(
            (
                r
                for r in hybrid_task_results
                if r["executor"] == "ThreadPoolExecutor"
                and r["extra_args"]
                and r["extra_args"].get("numpy_ratio") == ratio
            ),
            None,
        )

        if process_result and thread_result:
            process_time = process_result["total_time"]
            thread_time = thread_result["total_time"]
            speedup = process_time / thread_time if thread_time > 0 else float("inf")

            report.append(
                f"| {ratio:.2f} | {process_time:.3f} | {thread_time:.3f} | {speedup:.2f}x | {'ThreadPool' if speedup > 1 else 'ProcessPool'} |"
            )

    # 総合分析
    report.append(f"\n## 総合分析")

    # Python集中型タスクの総合分析
    python_results = [r for r in results if r["task_type"] == "python_intensive_task"]
    process_times = [
        r["total_time"]
        for r in python_results
        if r["executor"] == "ProcessPoolExecutor"
    ]
    thread_times = [
        r["total_time"] for r in python_results if r["executor"] == "ThreadPoolExecutor"
    ]

    if process_times and thread_times:
        avg_process = np.mean(process_times)
        avg_thread = np.mean(thread_times)
        speedup = avg_thread / avg_process if avg_process > 0 else float("inf")

        report.extend(
            [
                f"\n### 1. Python集中型タスク (GIL影響あり):",
                f"- ProcessPool平均時間: {avg_process:.3f}秒",
                f"- ThreadPool平均時間: {avg_thread:.3f}秒",
                f"- Thread/Process 速度比: {1 / speedup:.2f}x",
                f"- 結論: {'ThreadPoolExecutorが有利' if speedup < 1 else 'ProcessPoolExecutorが有利'}",
            ]
        )

    # NumPy最適化タスクの総合分析
    numpy_results = [r for r in results if r["task_type"] == "numpy_optimized_task"]
    process_times = [
        r["total_time"] for r in numpy_results if r["executor"] == "ProcessPoolExecutor"
    ]
    thread_times = [
        r["total_time"] for r in numpy_results if r["executor"] == "ThreadPoolExecutor"
    ]

    if process_times and thread_times:
        avg_process = np.mean(process_times)
        avg_thread = np.mean(thread_times)
        speedup = avg_process / avg_thread if avg_thread > 0 else float("inf")

        report.extend(
            [
                f"\n### 2. NumPy最適化タスク (GIL解放):",
                f"- ProcessPool平均時間: {avg_process:.3f}秒",
                f"- ThreadPool平均時間: {avg_thread:.3f}秒",
                f"- Process/Thread 速度比: {speedup:.2f}x",
                f"- 結論: {'ThreadPoolExecutorが有利' if speedup > 1 else 'ProcessPoolExecutorが有利'}",
            ]
        )

    # GILの影響に関する考察
    report.append(f"\n### GILの影響に関する考察")
    report.extend(
        [
            f"1. **Python集中型タスク**: GILの影響を強く受けるPythonコードでは、ThreadPoolExecutorの並列化効果は制限される傾向があります。"
            f"そのため、ProcessPoolExecutorのほうが有利なケースが多く見られます。",
            f"\n2. **NumPy最適化タスク**: NumPyの多くの演算はGILを解放するため、ThreadPoolExecutorでも並列化の恩恵を受けられます。"
            f"さらにプロセス間通信のオーバーヘッドがないため、ThreadPoolExecutorが有利になる場合があります。",
            f"\n3. **NumPy比率の影響**: ハイブリッドタスクにおいて、NumPy比率の増加に伴いThreadPoolExecutorの相対的優位性が高まる傾向が見られます。"
            f"これは、NumPy比率が高いほどGILの制約から解放され、スレッドベースの並列化が効果的になるためです。",
        ]
    )

    # スケーラビリティに関する考察
    report.append(f"\n### スケーラビリティに関する考察")
    report.extend(
        [
            f"- Python集中型タスクでは、ThreadPoolExecutorはワーカー数を増やしても性能向上が限られています。これはGILによる制約が原因です。",
            f"- NumPy最適化タスクでは、ThreadPoolExecutorはワーカー数の増加に伴って性能が向上する傾向があります。",
            f"- ProcessPoolExecutorは両方のタスクタイプでワーカー数の増加による性能向上が見られますが、プロセス生成と通信のオーバーヘッドがあります。",
        ]
    )

    # 結論
    report.append(f"\n## 結論")
    report.extend(
        [
            f"このベンチマークから、並列処理戦略を選択する際の一般的なガイドラインとして以下が挙げられます：",
            f"\n1. **CPU集中型のPure Pythonコード**: ProcessPoolExecutorを使用する",
            f"\n2. **NumPy/SciPyなどGILを解放する最適化された演算**: ThreadPoolExecutorを使用する（メモリ効率が良く、オーバーヘッドが少ない）",
            f"\n3. **混合コード**: コードの特性に応じて選択するか、NumPyなどの最適化された演算の比率を増やす工夫をする",
            f"\nGILは並列処理戦略を選ぶ上で重要な要素であり、タスクの特性を理解して適切なエグゼキュータを選択することが性能向上の鍵となります。",
        ]
    )

    # 生成された図のリンク
    report.extend(
        [
            f"\n## 生成されたグラフ",
            f"![GIL影響分析](../../fig/{BENCHMARK_NO}_{BENCHMARK_NAME}/gil_impact_analysis.png)",
            f"![スケーラビリティ分析](../../fig/{BENCHMARK_NO}_{BENCHMARK_NAME}/gil_scalability_analysis.png)",
        ]
    )

    return "\n".join(report)


def create_report_directory() -> Path:
    """レポートを保存するディレクトリを作成"""
    report_dir = Path("reports") / f"{BENCHMARK_NO}_{BENCHMARK_NAME}"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def save_report(report: str, report_dir: Path) -> None:
    """レポートをファイルに保存"""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"report_{now}.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"レポートを保存しました: {report_file}")


def main() -> None:
    """メイン実行関数"""
    print("===== GILの影響検証: ProcessPoolExecutor vs ThreadPoolExecutor =====")

    # システム情報
    cpu_count: int | None = os.cpu_count()
    mem_info = psutil.virtual_memory()
    print(f"システム情報:")
    print(f"- CPU: {cpu_count}コア")
    print(f"- メモリ: {mem_info.total / (1024**3):.1f} GB")

    # テスト設定
    data_size: int = 1000  # 1000x1000の行列
    worker_counts: list[int] = [1, 2, 4, 8]
    iterations: int = 3
    numpy_ratios: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]

    # テスト実行
    print("\nGIL影響テストを実行中...")
    results: list[dict[str, object]] = compare_gil_impact(
        data_size, worker_counts, iterations, numpy_ratios
    )

    # 結果の可視化
    print("\n結果を可視化中...")
    visualize_gil_impact(results)

    # 総合的な分析
    print("\n===== 総合分析 =====")

    # 1. Pythonコード（GIL影響あり）のテスト結果
    python_results = [r for r in results if r["task_type"] == "python_intensive_task"]

    if python_results:
        process_times = [
            r["total_time"]
            for r in python_results
            if r["executor"] == "ProcessPoolExecutor"
        ]
        thread_times = [
            r["total_time"]
            for r in python_results
            if r["executor"] == "ThreadPoolExecutor"
        ]

        if process_times and thread_times:
            avg_process = np.mean(process_times)
            avg_thread = np.mean(thread_times)
            speedup = avg_thread / avg_process if avg_process > 0 else float("inf")

            print(f"\n1. Pythonコード（GIL影響あり）:")
            print(f"   - ProcessPool平均時間: {avg_process:.3f}秒")
            print(f"   - ThreadPool平均時間: {avg_thread:.3f}秒")
            print(f"   - Thread/Process 速度比: {1 / speedup:.2f}x")
            print(
                f"   - 結論: {'ThreadPoolExecutorが有利' if speedup < 1 else 'ProcessPoolExecutorが有利'}"
            )

    # 2. NumPyコード（GIL解放）のテスト結果
    numpy_results = [r for r in results if r["task_type"] == "numpy_optimized_task"]

    if numpy_results:
        process_times = [
            r["total_time"]
            for r in numpy_results
            if r["executor"] == "ProcessPoolExecutor"
        ]
        thread_times = [
            r["total_time"]
            for r in numpy_results
            if r["executor"] == "ThreadPoolExecutor"
        ]

        if process_times and thread_times:
            avg_process = np.mean(process_times)
            avg_thread = np.mean(thread_times)
            speedup = avg_process / avg_thread if avg_thread > 0 else float("inf")

            print(f"\n2. NumPyコード（GIL解放）:")
            print(f"   - ProcessPool平均時間: {avg_process:.3f}秒")
            print(f"   - ThreadPool平均時間: {avg_thread:.3f}秒")
            print(f"   - Process/Thread 速度比: {speedup:.2f}x")
            print(
                f"   - 結論: {'ThreadPoolExecutorが有利' if speedup > 1 else 'ProcessPoolExecutorが有利'}"
            )

    # 3. ハイブリッドコードの分析
    hybrid_results: list[dict[str, object]] = [
        r for r in results if r["task_type"] == "hybrid_task"
    ]

    if hybrid_results:
        # NumPy比率ごとの分析
        numpy_ratios: list[float] = sorted(
            set(
                r["extra_args"]["numpy_ratio"]
                for r in hybrid_results
                if r["extra_args"]
            )
        )

        print("\n3. ハイブリッドコード（NumPy比率の影響）:")

        for ratio in numpy_ratios:
            process_result = next(
                (
                    r
                    for r in hybrid_results
                    if r["executor"] == "ProcessPoolExecutor"
                    and r["extra_args"]
                    and r["extra_args"]["numpy_ratio"] == ratio
                ),
                None,
            )
            thread_result = next(
                (
                    r
                    for r in hybrid_results
                    if r["executor"] == "ThreadPoolExecutor"
                    and r["extra_args"]
                    and r["extra_args"]["numpy_ratio"] == ratio
                ),
                None,
            )

            if process_result and thread_result:
                process_time: float = process_result["total_time"]
                thread_time: float = thread_result["total_time"]
                speedup: float = (
                    process_time / thread_time if thread_time > 0 else float("inf")
                )

                print(f"\n   NumPy比率 {ratio:.2f}:")
                print(f"   - ProcessPool時間: {process_time:.3f}秒")
                print(f"   - ThreadPool時間: {thread_time:.3f}秒")
                print(f"   - Process/Thread 速度比: {speedup:.2f}x")
                print(
                    f"   - 結論: {'ThreadPoolExecutorが有利' if speedup > 1 else 'ProcessPoolExecutorが有利'}"
                )

    # レポート生成
    print("\n===== レポート生成 =====")
    report = generate_markdown_report(
        results, data_size, worker_counts, iterations, numpy_ratios
    )
    report_dir = create_report_directory()
    save_report(report, report_dir)


# %%
if __name__ == "__main__":
    main()
