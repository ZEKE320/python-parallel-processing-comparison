from dataclasses import dataclass


@dataclass
class ResultStats:
    """処理結果の統計情報"""

    sum: float
    min: float
    max: float
    mean: float
    std: float


@dataclass
class TaskResult:
    """個別タスクの実行結果"""

    start_idx: int
    end_idx: int
    time: float
    result: ResultStats


@dataclass
class TestResult:
    """テスト実行の結果"""

    executor: str
    operation: str
    n_jobs: int
    total_time: float
    task_times: list[float]
    avg_task_time: float
    max_task_time: float
    min_task_time: float
    start_memory: float
    executor_memory: float
    end_memory: float
    memory_increase: float
    results: list[TaskResult]


@dataclass
class VisualizationInfo:
    """可視化で生成された画像ファイル情報"""

    timestamp: str
    time_comparison: str
    speedup_comparison: str
    memory_comparison: str
