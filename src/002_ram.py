# %%
import gc  # ガベージコレクタモジュールをインポート
import os
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime  # 日時情報の取得用に追加
from pathlib import Path  # pathlibをインポート

import japanize_matplotlib  # 日本語フォント設定
import matplotlib.pyplot as plt
import numpy as np
import psutil

# ベンチマーク識別子（定数）
BENCHMARK_NO = "002"
BENCHMARK_NAME = "ram"

# 演算タイプの日本語名マッピング（グローバル変数）
OPERATION_NAMES: dict[str, str] = {
    "matrix_product": "行列積",
    "svd": "特異値分解",
    "fft": "FFT変換",
    "stats": "統計計算",
}


def process_block(
    data: np.ndarray, block_id: int, block_size: int, operation_type: str
) -> dict[str, object]:
    """
    データブロックに対して指定された演算を実行する関数

    引数:
        data: 入力データ配列
        block_id: 処理するブロックのID
        block_size: ブロックのサイズ
        operation_type: 実行する演算の種類

    戻り値:
        処理結果の統計量と処理時間
    """
    # 配列全体からブロックの範囲を計算
    start_row: int = block_id * block_size
    end_row: int = min(start_row + block_size, data.shape[0])

    # 処理開始
    start_time: float = time.time()

    # 配列の一部を取得（ビュー）
    block_data: np.ndarray = data[start_row:end_row, :]

    # 演算タイプに応じた処理を実行
    if operation_type == "matrix_product":
        # 行列積（C実装、GILを解放）
        result: np.ndarray = np.dot(block_data, block_data.T)
    elif operation_type == "svd":
        # 特異値分解（計算負荷が高い演算）
        result: np.ndarray = np.linalg.svd(block_data, full_matrices=False)[0]
    elif operation_type == "fft":
        # 高速フーリエ変換（GILを解放）
        result: np.ndarray = np.fft.fft2(block_data)
    elif operation_type == "stats":
        # 統計量の計算（軽量な処理）
        result: np.ndarray = np.zeros((block_data.shape[0], 4))
        for i in range(block_data.shape[0]):
            row = block_data[i, :]
            result[i, 0] = np.mean(row)
            result[i, 1] = np.std(row)
            result[i, 2] = np.min(row)
            result[i, 3] = np.max(row)
    else:
        raise ValueError(f"不明な演算タイプです: {operation_type}")

    # 処理時間を計測
    proc_time: float = time.time() - start_time

    # メモリ使用量を取得する前にガベージコレクションを実行
    gc.collect()

    # メモリ使用量を取得
    process = psutil.Process(os.getpid())
    memory_used: float = process.memory_info().rss / (1024 * 1024)  # MB単位

    # 結果の概要を返す（大きな配列全体ではなく統計量のみ）
    result_summary: dict[str, object] = {
        "block_id": block_id,
        "rows_processed": end_row - start_row,
        "processing_time": proc_time,
        "memory_used": memory_used,
        "result_shape": result.shape,
        "result_stats": {
            "mean": float(np.mean(result)),
            "std": float(np.std(result)),
            "min": float(np.min(result)),
            "max": float(np.max(result)),
        },
    }

    # 関数終了前にガベージコレクションを実行
    gc.collect()

    return result_summary


def monitor_memory_usage(
    executor_class: type,
    data: np.ndarray,
    num_blocks: int,
    block_size: int,
    operation_type: str,
) -> dict[str, object]:
    """
    指定されたエグゼキュータで実行中のメモリ使用量を監視

    引数:
        executor_class: 使用するエグゼキュータクラス
        data: 処理する入力データ
        num_blocks: 処理するブロック数
        block_size: 各ブロックのサイズ
        operation_type: 実行する演算の種類

    戻り値:
        メモリ使用状況とパフォーマンス指標の辞書
    """
    executor_name: str = executor_class.__name__
    print(f"{executor_name}でメモリ使用量計測中（演算: {operation_type}）...")

    # メモリ測定前にガベージコレクションを実行
    gc.collect()

    # トレースマロックを開始
    tracemalloc.start()

    # 開始時のメモリ使用量を記録
    start_memory: float = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    _, start_peak = tracemalloc.get_traced_memory()

    # 時間計測開始
    start_time: float = time.time()

    # メモリ使用量の記録用
    memory_readings: list[float] = [start_memory]
    peak_readings: list[float] = [start_peak / (1024 * 1024)]  # MB単位に変換
    timestamps: list[float] = [0]

    # エグゼキュータを生成
    with executor_class(max_workers=num_blocks) as executor:
        # エグゼキュータ生成後、メモリ測定前にガベージコレクションを実行
        gc.collect()

        # エグゼキュータ生成後のメモリ使用量
        executor_memory: float = psutil.Process(os.getpid()).memory_info().rss / (
            1024 * 1024
        )
        memory_readings.append(executor_memory)
        _, peak = tracemalloc.get_traced_memory()
        peak_readings.append(peak / (1024 * 1024))
        timestamps.append(time.time() - start_time)

        # タスクをスケジュール
        futures = []
        for i in range(num_blocks):
            future = executor.submit(process_block, data, i, block_size, operation_type)
            futures.append(future)

        # 実行中のメモリ使用量を監視
        completed: int = 0
        while completed < num_blocks:
            # 完了したタスク数を確認
            completed = sum(1 for f in futures if f.done())

            # メモリ使用量を記録
            current_memory = psutil.Process(os.getpid()).memory_info().rss / (
                1024 * 1024
            )
            memory_readings.append(current_memory)

            _, peak = tracemalloc.get_traced_memory()
            peak_readings.append(peak / (1024 * 1024))

            timestamps.append(time.time() - start_time)

            # 少し待機
            time.sleep(0.1)

        # 結果を取得
        results = [future.result() for future in futures]

    # 処理終了後、メモリ測定前にガベージコレクションを実行
    gc.collect()

    # 処理終了後のメモリ使用量
    end_memory: float = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    memory_readings.append(end_memory)

    _, peak = tracemalloc.get_traced_memory()
    peak_readings.append(peak / (1024 * 1024))

    timestamps.append(time.time() - start_time)

    # 合計実行時間
    total_time: float = time.time() - start_time

    # タスクごとの処理時間統計
    task_times: list[float] = [result["processing_time"] for result in results]

    # トレースマロックを停止
    tracemalloc.stop()

    # 結果を返す前にガベージコレクションを実行
    gc.collect()

    # 結果をまとめる
    result: dict[str, object] = {
        "executor": executor_name,
        "operation": operation_type,
        "memory_readings": memory_readings,
        "peak_memory_readings": peak_readings,
        "timestamps": timestamps,
        "total_time": total_time,
        "task_times": task_times,
        "avg_task_time": float(np.mean(task_times)),
        "max_task_time": float(np.max(task_times)),
        "min_task_time": float(np.min(task_times)),
        "start_memory": start_memory,
        "max_memory": float(max(memory_readings)),
        "max_peak_memory": float(max(peak_readings)),
        "end_memory": end_memory,
        "memory_increase": float(max(memory_readings) - start_memory),
    }

    return result


def compare_memory_efficiency(
    data_size: int = 2000, num_blocks: int = 4, operations: list[str] | None = None
) -> list[dict[str, object]]:
    """
    様々な演算タイプに対するProcessPoolExecutorとThreadPoolExecutorのメモリ効率を比較

    引数:
        data_size: テストデータの行数・列数
        num_blocks: 処理ブロック数（並列度）
        operations: テストする演算タイプのリスト

    戻り値:
        テスト結果のリスト
    """
    if operations is None:
        operations = ["matrix_product", "svd", "fft", "stats"]

    # ベンチマーク開始前にガベージコレクションを実行
    gc.collect()

    # 共有データを作成
    print(f"{data_size}x{data_size}のテストデータを生成中...")
    data: np.ndarray = np.random.random((data_size, data_size))
    block_size: int = data_size // num_blocks

    # 大きなデータ作成後にガベージコレクションを実行
    gc.collect()

    # 結果格納用
    results: list[dict[str, object]] = []

    # 各演算タイプについてテスト
    for operation in operations:
        print(f"\n=== 演算タイプ: {operation} ===")

        # テスト前にガベージコレクションを実行
        gc.collect()

        # ProcessPoolExecutorの計測
        process_result: dict[str, object] = monitor_memory_usage(
            ProcessPoolExecutor, data, num_blocks, block_size, operation
        )
        results.append(process_result)

        # テスト間にガベージコレクションを実行
        gc.collect()

        # ThreadPoolExecutorの計測
        thread_result: dict[str, object] = monitor_memory_usage(
            ThreadPoolExecutor, data, num_blocks, block_size, operation
        )
        results.append(thread_result)

        # 結果を表示
        print(f"\n結果サマリー:")
        print(f"ProcessPoolExecutor:")
        print(f"- 合計実行時間: {process_result['total_time']:.3f}秒")
        print(f"- 最大メモリ使用量: {process_result['max_memory']:.1f} MB")
        print(f"- メモリ増加量: {process_result['memory_increase']:.1f} MB")

        print(f"\nThreadPoolExecutor:")
        print(f"- 合計実行時間: {thread_result['total_time']:.3f}秒")
        print(f"- 最大メモリ使用量: {thread_result['max_memory']:.1f} MB")
        print(f"- メモリ増加量: {thread_result['memory_increase']:.1f} MB")

        print(f"\n比較:")
        time_ratio: float = process_result["total_time"] / thread_result["total_time"]
        memory_ratio: float = (
            process_result["memory_increase"] / thread_result["memory_increase"]
            if thread_result["memory_increase"] > 0
            else float("inf")
        )

        print(f"- 時間比 (Process/Thread): {time_ratio:.2f}x")
        print(f"- メモリ増加比 (Process/Thread): {memory_ratio:.2f}x")

        # テスト後にガベージコレクションを実行
        gc.collect()

    return results


def visualize_memory_results(results: list[dict[str, object]]) -> None:
    """テスト結果を可視化"""
    # 可視化前にガベージコレクションを実行
    gc.collect()

    # 画像保存用ディレクトリを作成
    save_dir = Path("fig") / f"{BENCHMARK_NO}_{BENCHMARK_NAME}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 演算タイプごとに結果をグループ化
    operations: list[str] = sorted(set(r["operation"] for r in results))

    # 1. メモリ使用量の経時変化
    plt.figure(figsize=(15, 10))

    for i, operation in enumerate(operations):
        plt.subplot(2, 2, i + 1)

        for result in results:
            if result["operation"] == operation:
                if result["executor"] == "ProcessPoolExecutor":
                    plt.plot(
                        result["timestamps"],
                        result["memory_readings"],
                        "b-",
                        label="ProcessPool (RSS)",
                    )
                    plt.plot(
                        result["timestamps"],
                        result["peak_memory_readings"],
                        "b--",
                        label="ProcessPool (Peak)",
                    )
                else:
                    plt.plot(
                        result["timestamps"],
                        result["memory_readings"],
                        "r-",
                        label="ThreadPool (RSS)",
                    )
                    plt.plot(
                        result["timestamps"],
                        result["peak_memory_readings"],
                        "r--",
                        label="ThreadPool (Peak)",
                    )

        plt.xlabel("時間 (秒)")
        plt.ylabel("メモリ使用量 (MB)")
        plt.title(f"{OPERATION_NAMES.get(operation, operation)}のメモリ使用量")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "memory_usage_time_series.png", dpi=100)

    # 2. メモリ使用量と実行時間の比較
    plt.figure(figsize=(15, 5))

    # 2.1 メモリ増加量の比較
    plt.subplot(1, 2, 1)

    process_memory: list[float] = []
    thread_memory: list[float] = []
    op_labels: list[str] = []

    for operation in operations:
        process_result = next(
            (
                r
                for r in results
                if r["operation"] == operation
                and r["executor"] == "ProcessPoolExecutor"
            ),
            None,
        )
        thread_result = next(
            (
                r
                for r in results
                if r["operation"] == operation and r["executor"] == "ThreadPoolExecutor"
            ),
            None,
        )

        if process_result and thread_result:
            process_memory.append(float(process_result["memory_increase"]))
            thread_memory.append(float(thread_result["memory_increase"]))
            op_labels.append(OPERATION_NAMES.get(operation, operation))

    x = np.arange(len(op_labels))
    width: float = 0.35

    plt.bar(x - width / 2, process_memory, width, label="ProcessPool", color="skyblue")
    plt.bar(x + width / 2, thread_memory, width, label="ThreadPool", color="salmon")

    plt.xlabel("演算タイプ")
    plt.ylabel("メモリ増加量 (MB)")
    plt.title("エグゼキュータごとのメモリ増加量")
    plt.xticks(x, op_labels)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    # 2.2 実行時間の比較
    plt.subplot(1, 2, 2)

    process_times = []
    thread_times = []

    for operation in operations:
        process_result = next(
            (
                r
                for r in results
                if r["operation"] == operation
                and r["executor"] == "ProcessPoolExecutor"
            ),
            None,
        )
        thread_result = next(
            (
                r
                for r in results
                if r["operation"] == operation and r["executor"] == "ThreadPoolExecutor"
            ),
            None,
        )

        if process_result and thread_result:
            process_times.append(process_result["total_time"])
            thread_times.append(thread_result["total_time"])

    plt.bar(x - width / 2, process_times, width, label="ProcessPool", color="skyblue")
    plt.bar(x + width / 2, thread_times, width, label="ThreadPool", color="salmon")

    plt.xlabel("演算タイプ")
    plt.ylabel("実行時間 (秒)")
    plt.title("エグゼキュータごとの実行時間")
    plt.xticks(x, op_labels)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "memory_and_time_comparison.png", dpi=100)

    # 3. メモリ効率と時間効率のバブルチャート
    plt.figure(figsize=(10, 8))

    for operation in operations:
        process_result = next(
            (
                r
                for r in results
                if r["operation"] == operation
                and r["executor"] == "ProcessPoolExecutor"
            ),
            None,
        )
        thread_result = next(
            (
                r
                for r in results
                if r["operation"] == operation and r["executor"] == "ThreadPoolExecutor"
            ),
            None,
        )

        if process_result and thread_result:
            # ProcessPool用のバブル
            plt.scatter(
                process_result["memory_increase"],
                process_result["total_time"],
                s=200,
                alpha=0.6,
                color="blue",
                label=f"ProcessPool - {OPERATION_NAMES.get(operation, operation)}"
                if operation == operations[0]
                else "",
            )

            # ThreadPool用のバブル
            plt.scatter(
                thread_result["memory_increase"],
                thread_result["total_time"],
                s=200,
                alpha=0.6,
                color="red",
                label=f"ThreadPool - {OPERATION_NAMES.get(operation, operation)}"
                if operation == operations[0]
                else "",
            )

            # 演算タイプのラベル
            plt.annotate(
                OPERATION_NAMES.get(operation, operation),
                (
                    (
                        process_result["memory_increase"]
                        + thread_result["memory_increase"]
                    )
                    / 2,
                    (process_result["total_time"] + thread_result["total_time"]) / 2,
                ),
                ha="center",
            )

            # 同じ演算タイプの点を線で結ぶ
            plt.plot(
                [process_result["memory_increase"], thread_result["memory_increase"]],
                [process_result["total_time"], thread_result["total_time"]],
                "k--",
                alpha=0.3,
            )

    plt.xlabel("メモリ増加量 (MB)")
    plt.ylabel("実行時間 (秒)")
    plt.title("メモリ効率と時間効率のトレードオフ")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "memory_time_tradeoff.png", dpi=100)

    # 可視化完了後にメモリ開放
    plt.close("all")
    gc.collect()


def generate_markdown_report(
    results: list[dict[str, object]],
    data_size: int,
    num_blocks: int,
    operations: list[str],
) -> str:
    """テスト結果をMarkdownレポートとして生成

    引数:
        results: テスト結果のリスト
        data_size: テストデータのサイズ
        num_blocks: 処理ブロック数（並列度）
        operations: テストする演算タイプのリスト

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
        f"# メモリ効率の詳細比較: ProcessPoolExecutor vs ThreadPoolExecutor",
        f"\n## 実行概要",
        f"- ベンチマーク ID: {BENCHMARK_NO}_{BENCHMARK_NAME}",
        f"- 実行日時: {now}",
        f"\n## システム情報",
        f"- CPU: {cpu_count}コア",
        f"- メモリ: {mem_info.total / (1024**3):.1f} GB",
        f"\n## テスト設定",
        f"- データサイズ: {data_size}x{data_size} の行列",
        f"- 並列処理ブロック数: {num_blocks}",
        f"- 演算タイプ: {', '.join([OPERATION_NAMES.get(op, op) for op in operations])}",
    ]

    # テスト結果
    report.append(f"\n## 演算タイプごとの詳細結果")

    for operation in operations:
        op_name = OPERATION_NAMES.get(operation, operation)
        report.append(f"\n### {op_name}")

        # この演算タイプに関する結果を取得
        process_result = next(
            (
                r
                for r in results
                if r["operation"] == operation
                and r["executor"] == "ProcessPoolExecutor"
            ),
            None,
        )
        thread_result = next(
            (
                r
                for r in results
                if r["operation"] == operation and r["executor"] == "ThreadPoolExecutor"
            ),
            None,
        )

        if process_result and thread_result:
            process_time = process_result["total_time"]
            thread_time = thread_result["total_time"]
            time_ratio = process_time / thread_time if thread_time > 0 else float("inf")

            process_memory = process_result["memory_increase"]
            thread_memory = thread_result["memory_increase"]
            memory_ratio = (
                process_memory / thread_memory if thread_memory > 0 else float("inf")
            )

            report.extend(
                [
                    f"\n#### 実行時間",
                    f"- ProcessPoolExecutor: {process_time:.3f}秒",
                    f"- ThreadPoolExecutor: {thread_time:.3f}秒",
                    f"- 速度比 (Process/Thread): {time_ratio:.2f}x",
                    f"- 結論: {'ThreadPoolExecutorが有利' if time_ratio > 1.1 else 'ProcessPoolExecutorが有利' if time_ratio < 0.9 else '同等'}",
                    f"\n#### メモリ使用量",
                    f"- ProcessPoolExecutor 増加量: {process_memory:.1f} MB",
                    f"- ThreadPoolExecutor 増加量: {thread_memory:.1f} MB",
                    f"- メモリ比 (Process/Thread): {memory_ratio:.2f}x",
                    f"- 結論: {'ThreadPoolExecutorが有利' if memory_ratio > 1.1 else 'ProcessPoolExecutorが有利' if memory_ratio < 0.9 else '同等'}",
                    f"\n#### タスク実行統計 (ProcessPoolExecutor)",
                    f"- 平均タスク時間: {process_result['avg_task_time']:.3f}秒",
                    f"- 最大タスク時間: {process_result['max_task_time']:.3f}秒",
                    f"- 最小タスク時間: {process_result['min_task_time']:.3f}秒",
                    f"\n#### タスク実行統計 (ThreadPoolExecutor)",
                    f"- 平均タスク時間: {thread_result['avg_task_time']:.3f}秒",
                    f"- 最大タスク時間: {thread_result['max_task_time']:.3f}秒",
                    f"- 最小タスク時間: {thread_result['min_task_time']:.3f}秒",
                ]
            )

    # 総合分析
    report.append(f"\n## 総合分析")

    # 全体的な性能傾向を分析
    thread_advantage_ops = []
    process_advantage_ops = []
    memory_efficient_ops = []

    for operation in operations:
        process_result = next(
            (
                r
                for r in results
                if r["operation"] == operation
                and r["executor"] == "ProcessPoolExecutor"
            ),
            None,
        )
        thread_result = next(
            (
                r
                for r in results
                if r["operation"] == operation and r["executor"] == "ThreadPoolExecutor"
            ),
            None,
        )

        if process_result and thread_result:
            time_ratio = process_result["total_time"] / thread_result["total_time"]
            memory_ratio = (
                process_result["memory_increase"] / thread_result["memory_increase"]
                if thread_result["memory_increase"] > 0
                else float("inf")
            )

            op_name = OPERATION_NAMES.get(operation, operation)

            if time_ratio > 1.1:
                thread_advantage_ops.append(op_name)
            elif time_ratio < 0.9:
                process_advantage_ops.append(op_name)

            if memory_ratio > 1.1:
                memory_efficient_ops.append(op_name)

    report.extend(
        [
            f"\n### 実行時間の観点",
            f"- ThreadPoolExecutorが有利な演算: {', '.join(thread_advantage_ops) if thread_advantage_ops else 'なし'}",
            f"- ProcessPoolExecutorが有利な演算: {', '.join(process_advantage_ops) if process_advantage_ops else 'なし'}",
            f"\n### メモリ効率の観点",
            f"- ThreadPoolExecutorがメモリ効率の良い演算: {', '.join(memory_efficient_ops) if memory_efficient_ops else 'なし'}",
        ]
    )

    # 結論
    report.append(f"\n## 結論")

    if (
        len(thread_advantage_ops) > len(process_advantage_ops)
        and len(memory_efficient_ops) > 0
    ):
        report.append(
            f"このベンチマークでは、**ThreadPoolExecutor**がほとんどの演算で実行時間とメモリ効率の両面で優れています。"
        )
    elif len(process_advantage_ops) > len(thread_advantage_ops):
        report.append(
            f"このベンチマークでは、**ProcessPoolExecutor**が実行時間の面で優れていますが、メモリ使用量は多くなる傾向があります。"
        )
    else:
        report.append(
            f"このベンチマークでは、演算タイプによってパフォーマンス特性が異なります。メモリ効率を重視する場合はThreadPoolExecutorが、純粋な処理速度を重視する場合は演算タイプごとに適切なエグゼキュータを選択すべきです。"
        )

    # 生成された図のリンク
    report.extend(
        [
            f"\n## 生成されたグラフ",
            f"![メモリ使用量の経時変化](../../fig/{BENCHMARK_NO}_{BENCHMARK_NAME}/memory_usage_time_series.png)",
            f"![メモリ使用量と実行時間の比較](../../fig/{BENCHMARK_NO}_{BENCHMARK_NAME}/memory_and_time_comparison.png)",
            f"![メモリ効率と時間効率のトレードオフ](../../fig/{BENCHMARK_NO}_{BENCHMARK_NAME}/memory_time_tradeoff.png)",
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
    print("===== メモリ効率の詳細比較: ProcessPoolExecutor vs ThreadPoolExecutor =====")

    # システム情報取得前にガベージコレクション実行
    gc.collect()

    # システム情報
    cpu_count: int | None = os.cpu_count()
    mem_info = psutil.virtual_memory()
    print(f"システム情報:")
    print(f"- CPU: {cpu_count}コア")
    print(f"- メモリ: {mem_info.total / (1024**3):.1f} GB")

    # テスト実行
    data_size: int = 2000  # 2000x2000の行列
    num_blocks: int = 4  # 4並列
    operations: list[str] = ["matrix_product", "svd", "fft", "stats"]

    # メモリ効率比較
    results: list[dict[str, object]] = compare_memory_efficiency(
        data_size, num_blocks, operations
    )

    # 結果の可視化
    visualize_memory_results(results)

    # 総合結果と考察
    print("\n===== 総合結果 =====")

    # 各演算タイプごとの比較
    for operation in operations:
        process_result = next(
            (
                r
                for r in results
                if r["operation"] == operation
                and r["executor"] == "ProcessPoolExecutor"
            ),
            None,
        )
        thread_result = next(
            (
                r
                for r in results
                if r["operation"] == operation and r["executor"] == "ThreadPoolExecutor"
            ),
            None,
        )

        if process_result and thread_result:
            time_ratio: float = (
                process_result["total_time"] / thread_result["total_time"]
            )
            memory_ratio: float = (
                process_result["memory_increase"] / thread_result["memory_increase"]
                if thread_result["memory_increase"] > 0
                else float("inf")
            )

            op_name: str = OPERATION_NAMES.get(operation, operation)

            print(f"\n{op_name}:")
            print(
                f"- 実行時間: ProcessPool {process_result['total_time']:.3f}秒 vs ThreadPool {thread_result['total_time']:.3f}秒"
            )
            print(f"- 速度比 (Process/Thread): {time_ratio:.2f}x")
            print(
                f"- メモリ増加: ProcessPool {process_result['memory_increase']:.1f}MB vs ThreadPool {thread_result['memory_increase']:.1f}MB"
            )
            print(f"- メモリ比 (Process/Thread): {memory_ratio:.2f}x")

            if time_ratio > 1.1 and memory_ratio > 1.1:
                print(f"  → ThreadPoolExecutorが時間・メモリともに有利")
            elif time_ratio < 0.9 and memory_ratio < 0.9:
                print(f"  → ProcessPoolExecutorが時間・メモリともに有利")
            elif time_ratio > 1.1:
                print(f"  → ThreadPoolExecutorが時間効率で有利")
            elif memory_ratio > 1.1:
                print(f"  → ThreadPoolExecutorがメモリ効率で有利")
            else:
                print(f"  → 大きな差異なし")

    # レポート生成
    print("\n===== レポート生成 =====")
    report = generate_markdown_report(results, data_size, num_blocks, operations)
    report_dir = create_report_directory()
    save_report(report, report_dir)

    # 実行完了前に最終的なガベージコレクションを実行
    gc.collect()


# %%
if __name__ == "__main__":
    main()
