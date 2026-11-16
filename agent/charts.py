"""Simple charts from a query result."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agent.db import QueryResult  # noqa: E402


def draw_chart(res: QueryResult, kind: str, x: str, y: str, title: str, series: str | None,
               path: pathlib.Path) -> pathlib.Path:
    cols = res.columns
    for c in [x, y] + ([series] if series else []):
        if c not in cols:
            raise KeyError(f"column '{c}' not in result columns {cols}")
    ix, iy = cols.index(x), cols.index(y)
    groups: dict[str, list[tuple]] = {}
    for r in res.rows:
        key = str(r[cols.index(series)]) if series else y
        groups.setdefault(key, []).append((r[ix], r[iy]))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    n = len(groups)
    for i, (name, pts) in enumerate(groups.items()):
        pts = [(a, b) for a, b in pts if b is not None]
        xs, ys = [p[0] for p in pts], [float(p[1]) for p in pts]
        if kind == "line":
            ax.plot(xs, ys, marker="o", label=name)
        elif kind == "scatter":
            ax.scatter(xs, ys, label=name)
        elif kind == "bar":
            if n == 1:
                ax.bar([str(v) for v in xs], ys)
            else:
                width = 0.8 / n
                pos = range(len(xs))
                ax.bar([p + i * width for p in pos], ys, width=width, label=name)
                ax.set_xticks([p + 0.4 - width / 2 for p in pos], [str(v) for v in xs])
        else:
            raise ValueError(f"unknown chart kind {kind}")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.spines[["top", "right"]].set_visible(False)
    if n > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
