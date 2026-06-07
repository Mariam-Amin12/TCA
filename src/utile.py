from matplotlib import pyplot as plt


def plot_conversation(df, conv_id, tactic="Model Output"):
    fig, ax = plt.subplots(figsize=(10, 4))
    turn_ids = df["turn_id"].tolist()

    decision_color = {0: "#2ecc71", 1: "#e74c3c"}
    alpha_map      = {0: 0.10, 1: 0.18}
    for t, pred in zip(turn_ids, df["pred"]):
        ax.axvspan(t - 0.4, t + 0.4, color=decision_color[int(pred)], alpha=alpha_map[int(pred)])
    
    
    for col in df.columns:
        if col in ["conv_id", "turn_id", "pred"]:
            continue
        
        ax.plot(turn_ids, df[col], marker="o", label=col)

    ax.set_title(f"Conv {conv_id} | {tactic}", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05); ax.set_xticks(turn_ids)
    ax.set_xlabel("Turn"); ax.set_ylabel("Score"); ax.grid(True, alpha=0.25)

    line_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=col, markerfacecolor="#34495e", markersize=8)
        for col in df.columns if col not in ["conv_id", "turn_id", "pred"]
    ]
    patch_handles = [
        mpatches.Patch(color="#2ecc71", alpha=0.4, label="ALLOW (0)"),
        mpatches.Patch(color="#e74c3c", alpha=0.4, label="BLOCK (1)"),
    ]
    ax.legend(handles=line_handles + patch_handles, loc="upper left", fontsize=8)
    plt.tight_layout(); plt.show()
