import gc
import os

import numpy as np
import psutil

from .common import OPERATION_NAMES
from .models import TestResult, VisualizationInfo
from .parallel_test import compare_executors
from .reporting import create_report_directory, generate_markdown_report, save_report
from .visualization import visualize_results


# メイン実行関数
def main() -> None:
    """メイン実行関数"""
    print("===== NumPy並列処理: ProcessPoolExecutor vs ThreadPoolExecutor =====")

    # システム情報の表示前にガベージコレクション実行
    gc.collect()

    # システム情報の表示
    cpu_count: int | None = os.cpu_count()
    mem_info = psutil.virtual_memory()
    print("システム情報:")
    print(f"- CPU コア数: {cpu_count}")
    print(f"- メモリ: {mem_info.total / (1024**3):.1f} GB")

    # テスト設定
    data_size: int = 2000  # 2000x2000の行列
    n_jobs_list: list[int] = [2, 4, 8]  # 並列ジョブ数
    operations: list[str] = [
        "matrix_product",
        "fft",
        "element_wise",
        "python_loops",
    ]  # テストする演算

    # テスト実行
    results: list[TestResult] = compare_executors(data_size, n_jobs_list, operations)

    # 結果の可視化と画像ファイル情報の取得
    viz_info: VisualizationInfo = visualize_results(results)

    # 総合結果
    print("\n===== 総合結果 =====")

    # 各演算タイプごとの結論
    for operation in operations:
        operation_results = [r for r in results if r.operation == operation]
        process_times = [
            r.total_time
            for r in operation_results
            if r.executor == "ProcessPoolExecutor"
        ]
        thread_times = [
            r.total_time
            for r in operation_results
            if r.executor == "ThreadPoolExecutor"
        ]

        if process_times and thread_times:
            avg_process: float = float(np.mean(process_times))
            avg_thread: float = float(np.mean(thread_times))
            speedup: float = avg_process / avg_thread

            print(f"\n{OPERATION_NAMES.get(operation, operation)}:")
            print(f"- ProcessPool平均時間: {avg_process:.3f}秒")
            print(f"- ThreadPool平均時間: {avg_thread:.3f}秒")
            print(f"- 速度比: {speedup:.2f}x")

            if speedup > 1.1:
                print("  → ThreadPoolExecutorが有意に高速")
            elif speedup < 0.9:
                print("  → ProcessPoolExecutorが有意に高速")
            else:
                print("  → 有意な差なし")

    # レポート生成
    print("\n===== レポート生成 =====")
    report = generate_markdown_report(
        results, data_size, n_jobs_list, operations, viz_info
    )
    report_dir = create_report_directory()
    save_report(report, report_dir)

    # 実行完了前に最終的なガベージコレクションを実行
    gc.collect()


if __name__ == "__main__":
    main()
