from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError("pyarrow is required for Parquet output") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
