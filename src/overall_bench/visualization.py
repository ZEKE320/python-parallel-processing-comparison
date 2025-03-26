import gc
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from .common import BENCHMARK_NAME, BENCHMARK_NO, OPERATION_NAMES, get_project_root
from .models import TestResult, VisualizationInfo


# 結果の可視化
def visualize_results(results: list[TestResult]) -> VisualizationInfo:
    """
    テスト結果を可視化

    引数:
        results: テスト結果のリスト

    戻り値:
        VisualizationInfo: 生成された画像ファイル情報
    """
    # 可視化前にメモリをクリーンアップ
    gc.collect()

    # ファイル名に使用する日時文字列
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # プロジェクトルートを取得
    project_root = get_project_root()

    # 画像保存用ディレクトリを作成（プロジェクトルートからの相対パス）
    save_dir = project_root / "fig" / f"{BENCHMARK_NO}_{BENCHMARK_NAME}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 演算タイプと並列数で結果をグループ化
    grouped_results: dict[tuple[str, str], list[TestResult]] = {}
    for result in results:
        key = (result.operation, str(result.n_jobs))
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)

    # 各演算タイプごとにグラフを作成
    operations = sorted(set(result.operation for result in results))
    n_jobs_list = sorted(set(str(result.n_jobs) for result in results))

    # 1. 演算タイプごとの処理時間比較
    plt.figure(figsize=(15, 10))

    for i, operation in enumerate(operations):
        plt.subplot(2, 2, i + 1)

        # 各並列数ごとの処理時間をプロット
        process_times: list[float] = []
        thread_times: list[float] = []

        for n_jobs in n_jobs_list:
            key = (operation, n_jobs)
            if key in grouped_results and len(grouped_results[key]) == 2:
                for result in grouped_results[key]:
                    if result.executor == "ProcessPoolExecutor":
                        process_times.append(result.total_time)
                    else:
                        thread_times.append(result.total_time)

        # 並列数とエグゼキュータごとの棒グラフ
        bar_width: float = 0.35
        index = np.arange(len(n_jobs_list))

        plt.bar(
            index - bar_width / 2,
            process_times,
            bar_width,
            label="ProcessPool",
            color="skyblue",
        )
        plt.bar(
            index + bar_width / 2,
            thread_times,
            bar_width,
            label="ThreadPool",
            color="salmon",
        )

        plt.xlabel("並列ジョブ数")
        plt.ylabel("処理時間 (秒)")
        plt.title(f"{OPERATION_NAMES.get(operation, operation)}の処理時間比較")
        plt.xticks(index, n_jobs_list)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 各バーに処理時間を表示
        for j, v in enumerate(process_times):
            plt.text(j - bar_width / 2, v + 0.1, f"{v:.2f}s", ha="center")
        for j, v in enumerate(thread_times):
            plt.text(j + bar_width / 2, v + 0.1, f"{v:.2f}s", ha="center")

    # ファイル名を定義
    time_comparison_file = f"executor_time_comparison_{timestamp}.png"
    speedup_comparison_file = f"executor_speedup_comparison_{timestamp}.png"
    memory_comparison_file = f"executor_memory_comparison_{timestamp}.png"

    plt.tight_layout()
    plt.savefig(save_dir / time_comparison_file, dpi=100)

    # 2. 速度比の比較
    plt.figure(figsize=(10, 6))

    speedups: list[float] = []
    labels: list[str] = []

    for operation in operations:
        for n_jobs in n_jobs_list:
            key = (operation, n_jobs)
            if key in grouped_results and len(grouped_results[key]) == 2:
                process_time: float | None = None
                thread_time: float | None = None

                for result in grouped_results[key]:
                    if result.executor == "ProcessPoolExecutor":
                        process_time = result.total_time
                    else:
                        thread_time = result.total_time

                if process_time and thread_time:
                    speedup: float = process_time / thread_time
                    speedups.append(speedup)
                    labels.append(
                        f"{OPERATION_NAMES.get(operation, operation)}\n({n_jobs}ジョブ)"
                    )

    # スピードアップ率の棒グラフ
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, speedups)
    plt.axhline(y=1.0, color="r", linestyle="-", alpha=0.7)
    plt.ylabel("速度比 (Process時間 / Thread時間)")
    plt.title("演算タイプと並列数ごとの速度比較")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    # バーの色分け（1より大きい：ThreadPoolが有利、1より小さい：ProcessPoolが有利）
    for i, bar in enumerate(bars):
        if speedups[i] > 1:
            bar.set_color("green")  # ThreadPoolが有利
        else:
            bar.set_color("orange")  # ProcessPoolが有利
        plt.text(i, speedups[i] + 0.05, f"{speedups[i]:.2f}x", ha="center")

    plt.tight_layout()
    plt.savefig(save_dir / speedup_comparison_file, dpi=100)

    # 3. メモリ使用量の比較
    plt.figure(figsize=(12, 6))

    memory_increases: dict[str, list[float]] = {"process": [], "thread": []}
    mem_labels: list[str] = []

    for operation in operations:
        for n_jobs in n_jobs_list:
            key = (operation, n_jobs)
            if key in grouped_results and len(grouped_results[key]) == 2:
                for result in grouped_results[key]:
                    if result.executor == "ProcessPoolExecutor":
                        memory_increases["process"].append(result.memory_increase)
                    else:
                        memory_increases["thread"].append(result.memory_increase)

                mem_labels.append(
                    f"{OPERATION_NAMES.get(operation, operation)}\n({n_jobs}ジョブ)"
                )

    # メモリ使用量のグループ化棒グラフ
    bar_width = 0.35
    index = np.arange(len(mem_labels))

    plt.bar(
        index - bar_width / 2,
        memory_increases["process"],
        bar_width,
        label="ProcessPool",
        color="skyblue",
    )
    plt.bar(
        index + bar_width / 2,
        memory_increases["thread"],
        bar_width,
        label="ThreadPool",
        color="salmon",
    )

    plt.xlabel("演算タイプと並列数")
    plt.ylabel("メモリ増加量 (MB)")
    plt.title("エグゼキュータごとのメモリ使用量比較")
    plt.xticks(index, mem_labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / memory_comparison_file, dpi=100)

    # 可視化完了後にメモリ開放
    plt.close("all")
    gc.collect()

    # 生成したファイル名の情報を返す
    return VisualizationInfo(
        timestamp=timestamp,
        time_comparison=time_comparison_file,
        speedup_comparison=speedup_comparison_file,
        memory_comparison=memory_comparison_file,
    )
