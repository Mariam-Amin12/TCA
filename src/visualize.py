
from __future__ import annotations

from typing import List, Dict, Union, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd


# ── colour palette ────────────────────────────────────────────────────────────
_C = {
    "progressive":  "#2c3e50",
    "interaction":  "#378add",
    "pattern":      "#1D9E75",
    "cum_drift":    "#E24B4A",
    "drift_accel":  "#BA7517",
    "toxicity":     "#A32D2D",
    "threat":       "#BA7517",
    "topic_shift":  "#185FA5",
    "post_refusal": "#3B6D11",
    "allow_bg":     "#2ecc71",
    "block_bg":     "#e74c3c",
}


def _to_df(data: Union[List[Dict], pd.DataFrame]) -> pd.DataFrame:
    """Accept either a list of observe_turn() dicts or a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.reset_index(drop=True)

    rows = []
    for r in data:
        flat = {
            "turn_id":          r.get("turn_id", len(rows)),
            "prediction":       r.get("prediction", 0),
            "probability":      r.get("probability", 0.0),
            "progressive_risk": r.get("progressive_risk", 0.0),
            "interaction_risk": r.get("interaction_risk", 0.0),
            "pattern_risk":     r.get("pattern_risk", 0.0),
        }
        rf = r.get("raw_features", {})
        flat["toxicity_score"]     = rf.get("toxicity_score", 0.0)
        flat["threat_score"]       = rf.get("threat_score", 0.0)
        flat["topic_shift_score"]  = rf.get("topic_shift_score", 0.0)
        flat["cumulative_drift"]   = rf.get("cumulative_drift", 0.0)
        flat["drift_acceleration"] = rf.get("drift_acceleration", 0.0)
        flat["post_refusal"]       = rf.get("post_refusal", 0.0)
        rows.append(flat)
    return pd.DataFrame(rows)


def plot_conversation(
    data:       Union[List[Dict], pd.DataFrame],
    conv_id:    Union[int, str] = 0,
    title:      Optional[str]   = None,
    show_feats: bool            = True,
    threshold:  float           = 0.5,
    save_path:  Optional[str]   = None,
    figsize:    tuple           = (11, 8),
) -> plt.Figure:
    
    df = _to_df(data)
    turns = df["turn_id"].tolist()
    n = len(turns)

    any_attack = (df["prediction"] == 1).any()
    verdict    = "⚠  Attack detected" if any_attack else "✓  Clean conversation"

    nrows = 2 if show_feats else 1
    heights = [3, 2] if show_feats else [3]
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    if nrows == 1:
        axes = [axes]
    ax1, *rest = axes
    ax2 = rest[0] if rest else None

    # ── shared helper: draw allow/block column shading ────────────────────
    def _shade(ax):
        half = 0.4 if n > 1 else 0.45
        for t, pred in zip(turns, df["prediction"]):
            color = _C["block_bg"] if pred == 1 else _C["allow_bg"]
            alpha = 0.14 if pred == 1 else 0.07
            ax.axvspan(t - half, t + half, color=color, alpha=alpha, linewidth=0)

    # ══ subplot 1: risk scores ════════════════════════════════════════════
    _shade(ax1)

    ax1.plot(turns, df["progressive_risk"],  "D-",  color=_C["progressive"],
             linewidth=2.5, markersize=7, label="progressive risk",  zorder=5)
    ax1.plot(turns, df["interaction_risk"],  "o-",  color=_C["interaction"],
             linewidth=1.8, markersize=5, label="interaction risk",  zorder=4)
    ax1.plot(turns, df["pattern_risk"],      "s-",  color=_C["pattern"],
             linewidth=1.8, markersize=5, label="pattern risk",      zorder=4)
    ax1.plot(turns, df["cumulative_drift"],  "^--", color=_C["cum_drift"],
             linewidth=1.2, markersize=5, alpha=0.85, label="cumulative drift", zorder=3)
    ax1.plot(turns, df["drift_acceleration"],"x--", color=_C["drift_accel"],
             linewidth=1.2, markersize=6, alpha=0.85, label="drift accel",      zorder=3)

    # probability as a faint filled area
    ax1.fill_between(turns, df["probability"], alpha=0.08, color=_C["progressive"],
                     label="_nolegend_")
    ax1.plot(turns, df["probability"], "-",  color=_C["progressive"],
             linewidth=0.8, alpha=0.4, linestyle="dotted", label="probability")

    # threshold line
    ax1.axhline(threshold, color="#888", linewidth=0.8, linestyle=":",
                label=f"threshold ({threshold})")

    ax1.set_ylim(-0.02, 1.08)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.grid(True, alpha=0.18, linewidth=0.6)
    ax1.set_xticks(turns)
    ax1.tick_params(axis="both", labelsize=9)

   
    line_handles = [
        mlines.Line2D([], [], color=_C["progressive"], marker="D",
                      linewidth=2, markersize=6, label="progressive risk"),
        mlines.Line2D([], [], color=_C["interaction"], marker="o",
                      linewidth=1.6, markersize=5, label="interaction risk"),
        mlines.Line2D([], [], color=_C["pattern"],     marker="s",
                      linewidth=1.6, markersize=5, label="pattern risk"),
        mlines.Line2D([], [], color=_C["cum_drift"],   marker="^",
                      linestyle="--", linewidth=1.2, markersize=5, label="cumulative drift"),
        mlines.Line2D([], [], color=_C["drift_accel"], marker="x",
                      linestyle="--", linewidth=1.2, markersize=5, label="drift accel"),
        mlines.Line2D([], [], color=_C["progressive"], linestyle="dotted",
                      linewidth=0.9, alpha=0.6, label="probability"),
        mlines.Line2D([], [], color="#888", linestyle=":",
                      linewidth=0.9, label=f"threshold ({threshold})"),
    ]
    patch_handles = [
        mpatches.Patch(color=_C["allow_bg"], alpha=0.35, label="ALLOW (0)"),
        mpatches.Patch(color=_C["block_bg"], alpha=0.35, label="BLOCK (1)"),
    ]
    ax1.legend(
        handles=line_handles + patch_handles,
        loc="upper left", fontsize=8, ncol=2,
        framealpha=0.85, edgecolor="#ccc",
    )

   
    if any_attack:
        first = df.index[df["prediction"] == 1][0]
        ft    = turns[first]
        ax1.annotate(
            "first\ndetection",
            xy=(ft, df.loc[first, "progressive_risk"]),
            xytext=(ft + 0.3, min(df.loc[first, "progressive_risk"] + 0.18, 1.0)),
            fontsize=8, color=_C["block_bg"],
            arrowprops=dict(arrowstyle="->", color=_C["block_bg"], lw=1.2),
        )

    if ax2 is not None:
        _shade(ax2)

        ax2.plot(turns, df["toxicity_score"],     "o-",  color=_C["toxicity"],
                 linewidth=1.5, markersize=4, label="toxicity")
        ax2.plot(turns, df["threat_score"],       "s-",  color=_C["threat"],
                 linewidth=1.5, markersize=4, label="threat")
        ax2.plot(turns, df["topic_shift_score"],  "^-",  color=_C["topic_shift"],
                 linewidth=1.5, markersize=4, label="topic shift")
        ax2.plot(turns, df["drift_acceleration"], "x--", color=_C["drift_accel"],
                 linewidth=1.2, markersize=5, alpha=0.85, label="drift accel")
        ax2.plot(turns, df["post_refusal"],       "D:",  color=_C["post_refusal"],
                 linewidth=1.2, markersize=5, label="post-refusal")

        ax2.set_ylim(-0.05, 1.08)
        ax2.set_ylabel("Raw features", fontsize=10)
        ax2.set_xlabel("Turn", fontsize=10)
        ax2.set_xticks(turns)
        ax2.set_xticklabels([f"T{t}" for t in turns], fontsize=9)
        ax2.grid(True, alpha=0.18, linewidth=0.6)
        ax2.tick_params(axis="y", labelsize=9)
        ax2.legend(loc="upper left", fontsize=8, ncol=3,
                   framealpha=0.85, edgecolor="#ccc")
    else:
        ax1.set_xlabel("Turn", fontsize=10)
        ax1.set_xticklabels([f"T{t}" for t in turns], fontsize=9)

    auto_title = title or f"Conv {conv_id}  |  {verdict}"
    color = "#c0392b" if any_attack else "#1D9E75"
    fig.suptitle(auto_title, fontsize=12, fontweight="bold", color=color, y=1.01)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
    return fig


def plot_conversations(
    conversations: List[Union[List[Dict], pd.DataFrame]],
    conv_ids:      Optional[List] = None,
    save_dir:      Optional[str]  = None,
    **kwargs,
) -> List[plt.Figure]:
   
    ids  = conv_ids or list(range(len(conversations)))
    figs = []
    for cid, data in zip(ids, conversations):
        sp = f"{save_dir}/conv_{cid}.png" if save_dir else None
        fig = plot_conversation(data, conv_id=cid, save_path=sp, **kwargs)
        figs.append(fig)
    return figs

