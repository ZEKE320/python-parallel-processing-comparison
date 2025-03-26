from pathlib import Path

import git

# ベンチマーク識別子（定数）
BENCHMARK_NO = "001"
BENCHMARK_NAME = "overall"

# 演算タイプの日本語名マッピング（グローバル変数）
OPERATION_NAMES: dict[str, str] = {
    "matrix_product": "行列積",
    "fft": "FFT変換",
    "element_wise": "要素ごとの演算",
    "python_loops": "Pythonループ",
}


def get_project_root() -> Path:
    """
    プロジェクトのルートディレクトリを取得する

    戻り値:
        Path: プロジェクトルートディレクトリのパス
    """
    # gitpythonが利用可能な場合はリポジトリのルートを取得
    repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
    return Path(repo.git.rev_parse("--show-toplevel"))
