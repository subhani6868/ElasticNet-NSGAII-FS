from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

try:
    from scipy import sparse  # type: ignore
except Exception:  # pragma: no cover
    sparse = None  # type: ignore

def _as_binary_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    uniq = np.unique(y)
    if set(uniq.tolist()) == {-1, 1}:
        return ((y + 1) // 2).astype(int)
    if set(uniq.tolist()) == {0, 1}:
        return y.astype(int)
    if len(uniq) == 2:
        return (y == uniq.max()).astype(int)
    return y.astype(int)

def _read_labels(p: Path) -> np.ndarray:
    txt = p.read_text(encoding="utf-8", errors="ignore").strip().split()
    y = np.array([float(t) for t in txt], dtype=float)
    return _as_binary_labels(y)

def _read_dense_data_whitespace(p: Path) -> np.ndarray:
    # NIPS 2003 dense .data files are whitespace-separated
    df = pd.read_csv(p, header=None, sep=r"\s+", engine="python")
    return df.to_numpy(dtype=float)

def _parse_sparse_lines(p: Path, n_features: Optional[int] = None, one_based: bool = True) -> np.ndarray:
    """Parse sparse data lines for Dexter/Dorothea.
    Each line can contain:
      - index:value tokens (e.g., 12:0.3 140:1)
      - OR binary index tokens (e.g., 3 10 21 1005)
    Indices are typically 1-based in challenge files.
    """
    if sparse is None:
        raise ImportError("scipy is required for parsing sparse datasets (Dexter/Dorothea). Install scipy>=1.10.")
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    max_idx = -1

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for r, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            for tok in line.split():
                if ":" in tok:
                    i_str, v_str = tok.split(":", 1)
                    c = int(i_str)
                    v = float(v_str)
                else:
                    c = int(tok)
                    v = 1.0
                if one_based:
                    c -= 1
                rows.append(r)
                cols.append(c)
                data.append(v)
                if c > max_idx:
                    max_idx = c

    d = int(n_features) if n_features is not None else (max_idx + 1)
    n_rows = (max(rows) + 1) if rows else 0
    X = sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, d), dtype=float)
    return X.toarray()

def _find_any(dirp: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        for p in dirp.glob(pat):
            if p.exists():
                return p
    return None

def load_nips2003_dataset(raw_dir: Path, dataset_key: str) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    key = dataset_key.lower()

    def dense_pair(prefix: str):
        Xtr = _find_any(raw_dir, [f"{prefix}_train.data", f"{prefix}.train.data", f"{prefix}_train.csv", f"{prefix}.train"])
        ytr = _find_any(raw_dir, [f"{prefix}_train.labels", f"{prefix}.train.labels", f"{prefix}_train.y"])
        Xva = _find_any(raw_dir, [f"{prefix}_valid.data", f"{prefix}.valid.data", f"{prefix}_valid.csv", f"{prefix}.valid"])
        yva = _find_any(raw_dir, [f"{prefix}_valid.labels", f"{prefix}.valid.labels", f"{prefix}_valid.y"])
        return Xtr, ytr, Xva, yva

    if key in {"arcene", "gisette", "madelon"}:
        prefix = key
        Xtr_p, ytr_p, Xva_p, yva_p = dense_pair(prefix)
        if Xtr_p is None:
            Xtr_p, ytr_p, Xva_p, yva_p = dense_pair(prefix.capitalize())

        if Xtr_p is None or ytr_p is None:
            raise FileNotFoundError(
                f"Missing train files for {key}. Expected '{prefix}_train.data' and '{prefix}_train.labels' under {raw_dir}"
            )

        Xtr = _read_dense_data_whitespace(Xtr_p)
        ytr = _read_labels(ytr_p)

        if Xva_p is not None and yva_p is not None:
            Xva = _read_dense_data_whitespace(Xva_p)
            yva = _read_labels(yva_p)
            X = np.vstack([Xtr, Xva])
            y = np.concatenate([ytr, yva])
        else:
            X, y = Xtr, ytr
        return X, y, None

    if key in {"dexter", "dorothea"}:
        prefix = key
        Xtr_p = _find_any(raw_dir, [f"{prefix}_train.data", f"{prefix}.train.data", f"{prefix}_train.txt", f"{prefix}.train"])
        ytr_p = _find_any(raw_dir, [f"{prefix}_train.labels", f"{prefix}.train.labels", f"{prefix}_train.labels.txt"])
        Xva_p = _find_any(raw_dir, [f"{prefix}_valid.data", f"{prefix}.valid.data", f"{prefix}_valid.txt", f"{prefix}.valid"])
        yva_p = _find_any(raw_dir, [f"{prefix}_valid.labels", f"{prefix}.valid.labels", f"{prefix}_valid.labels.txt"])

        if Xtr_p is None or ytr_p is None:
            raise FileNotFoundError(
                f"Missing train files for {key}. Expected '{prefix}_train.data' and '{prefix}_train.labels' under {raw_dir}"
            )

        Xtr = _parse_sparse_lines(Xtr_p, n_features=None, one_based=True)
        ytr = _read_labels(ytr_p)

        if Xva_p is not None and yva_p is not None:
            Xva = _parse_sparse_lines(Xva_p, n_features=Xtr.shape[1], one_based=True)
            yva = _read_labels(yva_p)
            X = np.vstack([Xtr, Xva])
            y = np.concatenate([ytr, yva])
        else:
            X, y = Xtr, ytr
        return X, y, None

    raise ValueError(f"Unsupported NIPS 2003 dataset key: {dataset_key}")
