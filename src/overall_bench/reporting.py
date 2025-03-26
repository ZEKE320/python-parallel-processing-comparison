import os
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

from .common import BENCHMARK_NAME, BENCHMARK_NO, OPERATION_NAMES, get_project_root
from .models import TestResult, VisualizationInfo


def generate_markdown_report(
    results: list[TestResult],
    data_size: int,
    n_jobs_list: list[int],
    operations: list[str],
    visualization_info: VisualizationInfo,
) -> str:
    """テスト結果をMarkdownレポートとして生成

    引数:
        results: テスト結果のリスト
        data_size: テストデータのサイズ
        n_jobs_list: 並列ジョブ数のリスト
        operations: テストした演算タイプのリスト
        visualization_info: 可視化で生成された画像ファイル情報

    戻り値:
        str: Markdownフォーマットのレポート
    """
    # システム情報
    cpu_count: int | None = os.cpu_count()
    mem_info = psutil.virtual_memory()

    # 画像ファイル名に使用するタイムスタンプ
    timestamp = visualization_info.timestamp

    # レポートのヘッダー
    report = [
        "# NumPy並列処理ベンチマーク: ProcessPoolExecutor vs ThreadPoolExecutor",
        "\n## 実行概要",
        f"- ベンチマーク ID: {BENCHMARK_NO}_{BENCHMARK_NAME}",
        f"- 実行日時: {timestamp}",
        "\n## システム情報",
        f"- CPU: {cpu_count}コア",
        f"- メモリ: {mem_info.total / (1024**3):.1f} GB",
        "\n## テスト設定",
        f"- データサイズ: {data_size}x{data_size} の行列",
        f"- 並列ジョブ数: {', '.join(map(str, n_jobs_list))}",
        f"- 演算タイプ: {', '.join([OPERATION_NAMES.get(op, op) for op in operations])}",
    ]

    # テスト結果
    report.append("\n## テスト結果")

    for operation in operations:
        op_results = [r for r in results if r.operation == operation]
        report.append(f"\n### {OPERATION_NAMES.get(operation, operation)}")

        # 各並列数ごとの結果
        for n_jobs in n_jobs_list:
            job_results = [r for r in op_results if r.n_jobs == n_jobs]
            process_result = next(
                (r for r in job_results if r.executor == "ProcessPoolExecutor"), None
            )
            thread_result = next(
                (r for r in job_results if r.executor == "ThreadPoolExecutor"), None
            )

            if process_result and thread_result:
                process_time = process_result.total_time
                thread_time = thread_result.total_time
                speedup = (
                    process_time / thread_time if thread_time > 0 else float("inf")
                )

                report.extend(
                    [
                        f"\n#### 並列ジョブ数: {n_jobs}",
                        f"- ProcessPoolExecutor 実行時間: {process_time:.3f}秒",
                        f"- ThreadPoolExecutor 実行時間: {thread_time:.3f}秒",
                        f"- 速度比 (Process/Thread): {speedup:.2f}x",
                        f"- 結論: {'ThreadPoolExecutorが有利' if speedup > 1 else 'ProcessPoolExecutorが有利' if speedup < 1 else '同等'}",
                        "\n**メモリ使用量:**",
                        f"- ProcessPoolExecutor: {process_result.memory_increase:.1f} MB",
                        f"- ThreadPoolExecutor: {thread_result.memory_increase:.1f} MB",
                    ]
                )

    # 総合分析
    report.append("\n## 総合分析")

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

            report.extend(
                [
                    f"\n### {OPERATION_NAMES.get(operation, operation)}:",
                    f"- ProcessPool平均時間: {avg_process:.3f}秒",
                    f"- ThreadPool平均時間: {avg_thread:.3f}秒",
                    f"- 速度比: {speedup:.2f}x",
                ]
            )

            if speedup > 1.1:
                report.append("- 結論: ThreadPoolExecutorが有意に高速")
            elif speedup < 0.9:
                report.append("- 結論: ProcessPoolExecutorが有意に高速")
            else:
                report.append("- 結論: 両者の性能差は有意ではない")

    # 生成された図のリンク
    report.extend(
        [
            "\n## 生成されたグラフ",
            f"![演算タイプごとの処理時間比較](../../fig/{BENCHMARK_NO}_{BENCHMARK_NAME}/{visualization_info.time_comparison})",
            f"![速度比の比較](../../fig/{BENCHMARK_NO}_{BENCHMARK_NAME}/{visualization_info.speedup_comparison})",
            f"![メモリ使用量の比較](../../fig/{BENCHMARK_NO}_{BENCHMARK_NAME}/{visualization_info.memory_comparison})",
        ]
    )

    return "\n".join(report)


def create_report_directory() -> Path:
    """レポートを保存するディレクトリを作成"""
    # プロジェクトルートを取得
    project_root = get_project_root()

    # レポートディレクトリをプロジェクトルートからの相対パスで作成
    report_dir = project_root / "reports" / f"{BENCHMARK_NO}_{BENCHMARK_NAME}"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def save_report(report: str, report_dir: Path) -> None:
    """レポートをファイルに保存"""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"report_{now}.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"レポートを保存しました: {report_file}")
