"""Figures for plan A (final version): overview schematic (PDF + editable SVG), trajectories,
local rule / first contact, and shock-persistence plane / trade-off.

Reads paper/scripts/numbers.json. Run after compute_numbers.py:
    python paper/scripts/make_figures_planA.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
N = json.loads((HERE / "numbers.json").read_text(encoding="utf-8"))

MODELS = ["Qwen2.5-7B", "Qwen3-14B", "Qwen3-30B", "GLM-4-9B", "Llama-3.1-8B"]
MCOL = {"Qwen2.5-7B": "#4477AA", "Qwen3-14B": "#228833", "Qwen3-30B": "#C99A00", "GLM-4-9B": "#AA3377", "Llama-3.1-8B": "#33A9C4"}
MMARK = {"Qwen2.5-7B": "o", "Qwen3-14B": "s", "Qwen3-30B": "^", "GLM-4-9B": "D", "Llama-3.1-8B": "v"}
RED, BLUE, GREY, YEL, INK = "#C0392B", "#1F618D", "#6E7B7C", "#F2C94C", "#222222"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.2, "axes.labelsize": 6.8, "xtick.labelsize": 6.3,
                     "ytick.labelsize": 6.3, "legend.fontsize": 6.2, "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
                     "xtick.major.size": 2.3, "ytick.major.size": 2.3, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none", "figure.dpi": 150})


def save(fig, name, tight=True, svg=False):
    kw = dict(bbox_inches="tight", pad_inches=.015) if tight else {}
    fig.savefig(OUT / f"{name}.pdf", **kw)
    fig.savefig(OUT / f"{name}.png", dpi=300, **kw)
    if svg:
        fig.savefig(OUT / f"{name}.svg", **kw)
    plt.close(fig)


def eff(model, grp, sched, arm):
    return N["effects"].get(f"{model}|{grp}|{sched}|{arm}")


# =========================================================================== Figure 1: overview
W, H = 7.05, 1.60


def box(ax, x, y, w, h, fc="white", ec=INK, lw=.7, r=.05, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, a, b, color=INK, lw=.8, style="-|>", rad=0.0, z=3, ls="-", ms=6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def fig_overview():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def title(x, text):
        ax.text(x, H - .04, text, fontsize=7.2, fontweight="bold", va="top", ha="left", color=INK)

    # ---------------- (a) pulse perturbation
    x0 = .05
    title(x0, "(a) Transient perturbation of an agent network")
    g = nx.watts_strogatz_graph(24, 4, 0.18, seed=5)
    cx, cy, rx, ry = x0 + .55, .89, .50, .37
    ang = np.pi / 2 - 2 * np.pi * np.arange(24) / 24
    pos = {i: (cx + rx * np.cos(a), cy + ry * np.sin(a)) for i, a in enumerate(ang)}
    for u, v in g.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#B5BDBE", lw=.45, zorder=1)
    seeds = {3, 12, 19}
    for i, (x, y) in pos.items():
        ax.scatter(x, y, s=20 if i in seeds else 13, color=RED if i in seeds else "white",
                   edgecolor=RED if i in seeds else "#4D5656", lw=.6, zorder=3)
    tx = x0 + 1.20
    ax.scatter(tx + .04, 1.12, s=20, color=RED, zorder=3)
    ax.text(tx + .13, 1.12, "3 seed agents send\na false message in\nrounds 2–3, then stop", va="center", ha="left", linespacing=1.05, fontsize=6.1)
    ax.scatter(tx + .04, .72, s=13, color="white", edgecolor="#4D5656", lw=.6, zorder=3)
    ax.text(tx + .13, .72, "21 other agents:\ntheir error is the\noutcome", va="center", ha="left", linespacing=1.05, fontsize=6.1)
    tw, ty, th = .215, .17, .17
    xs = x0 + .05 + np.arange(9) * (tw + .012)
    for t, x in enumerate(xs):
        fc = YEL if t in (2, 3) else ("#D9DDDD" if t >= 6 else "white")
        ax.add_patch(Rectangle((x, ty), tw, th, fc=fc, ec="#4D5656", lw=.5, zorder=2))
        ax.text(x + tw / 2, ty + th / 2, str(t), ha="center", va="center", fontsize=6)
    ax.text(xs[0], ty - .03, "round", ha="left", va="top", fontsize=5.8, color="#555")
    ax.text((xs[2] + xs[3] + tw) / 2, ty - .03, "perturbation", ha="center", va="top", fontsize=5.8, color="#8a6d00")
    ax.text((xs[6] + xs[8] + tw) / 2, ty - .03, "outcome window", ha="center", va="top", fontsize=5.8, color="#444")

    # ---------------- (b) agent inputs and the two structural factors
    x0 = 2.36
    title(x0, "(b) Inputs of an agent and two structural factors")
    cw, ch = 1.60, .27
    ys = [1.02, .69, .36]
    chips = [("private signal (correct with prob. 0.7)", "signal access: one-time ($\\rho{=}0$) / recurrent ($\\rho{=}1$)", BLUE, "#EAF2F8"),
             ("its own previous answer", "nothing else is carried over between rounds", "#777777", "#F4F6F6"),
             ("messages of up to 4 neighbours", "peer relay: intact or blocked", RED, "#FDEDEC")]
    ax_x = x0 + cw + .26
    aw, ah = .60, .40
    ay = ys[0] + ch - ah - .02
    for (t1, t2, c, fc), y in zip(chips, ys):
        box(ax, x0, y, cw, ch, fc=fc, ec=c, lw=.8)
        ax.text(x0 + .05, y + ch * .68, t1, va="center", ha="left", fontsize=6.1, color=INK)
        ax.text(x0 + .05, y + ch * .26, t2, va="center", ha="left", fontsize=5.0, color=c, style="italic")
    box(ax, ax_x, ay, aw, ah, fc="#FFFFFF", ec=INK, lw=.9, r=.07)
    ax.text(ax_x + aw / 2, ay + ah / 2, "LLM agent", ha="center", va="center", fontsize=6.4, fontweight="bold")
    arrow(ax, (x0 + cw + .02, ys[0] + ch / 2), (ax_x - .02, ay + ah * .72), color=BLUE)
    arrow(ax, (x0 + cw + .02, ys[1] + ch / 2), (ax_x - .02, ay + ah * .40), color="#777777")
    arrow(ax, (x0 + cw + .02, ys[2] + ch / 2), (ax_x + .06, ay - .02), color=RED)
    mx = ax_x + aw / 2 + .08
    arrow(ax, (mx, ay - .02), (mx, ay - .17))
    ax.text(mx, ay - .19, "public message\n+ answer (A or B)", ha="center", va="top", fontsize=5.9, linespacing=1.03)
    ly = .11
    ax.plot([mx, mx], [ay - .45, ly], color=RED, lw=.8, ls=(0, (3, 1.5)), zorder=2)
    ax.plot([mx, x0 + .3], [ly, ly], color=RED, lw=.8, ls=(0, (3, 1.5)), zorder=2)
    arrow(ax, (x0 + .3, ly), (x0 + .3, ys[2] - .02), color=RED, ls=(0, (3, 1.5)))
    ax.text(x0 + .40, ly + .03, "peer relay: it is a neighbour's input in the next round", ha="left", va="bottom",
            fontsize=5.2, color=RED, style="italic")

    # ---------------- (c) shock vs persistence
    x0 = 5.16
    title(x0, "(c) Shock and persistence")
    a = fig.add_axes([(x0 + .15) / W, .30 / H, .92 / W, .96 / H])
    t = np.arange(9)
    red = np.array([0, 0, .5, 1.0, 1.02, 1.06, 1.08, 1.1, 1.1])
    blue = np.array([0, 0, .5, 1.0, .62, .34, .16, .06, .02])
    a.axvspan(1.5, 3.5, color=YEL, alpha=.35, lw=0)
    a.axvspan(5.5, 8.3, color="#000000", alpha=.06, lw=0)
    a.plot(t, red, color=RED, lw=1.5)
    a.plot(t, blue, color=BLUE, lw=1.5)
    a.set_xlim(0, 8.3)
    a.set_ylim(-.06, 1.35)
    a.set_xticks([0, 2, 4, 6, 8])
    a.set_yticks([])
    a.tick_params(labelsize=5.6, length=2, width=.5, pad=1.5)
    a.set_xlabel("round", fontsize=5.8, labelpad=0)
    a.set_ylabel("excess error", fontsize=5.8, labelpad=2)
    a.annotate("", xy=(3, 1.0), xytext=(3, 0), arrowprops=dict(arrowstyle="<->", lw=.6, color=INK, shrinkA=0, shrinkB=0))
    a.text(2.85, .5, "shock", ha="right", va="center", fontsize=5.8, rotation=90)
    a.annotate("", xy=(7, 1.08), xytext=(7, .08), arrowprops=dict(arrowstyle="<->", lw=.6, color=INK, shrinkA=0, shrinkB=0))
    a.text(7.2, .58, "persistence", ha="left", va="center", fontsize=5.8, rotation=90)
    a.text(3.9, 1.21, "persists", color=RED, fontsize=5.8, ha="left", va="center")
    a.text(4.75, .50, "decays", color=BLUE, fontsize=5.8, ha="left", va="center")
    tx = x0 + 1.17
    ax.text(tx, 1.15, "Shock", fontsize=6.0, fontweight="bold", va="center")
    ax.text(tx, 1.06, "excess error\nwhen the false\nmessages stop", fontsize=5.5, va="top", linespacing=1.05)
    ax.text(tx, .62, "Persistence", fontsize=6.0, fontweight="bold", va="center")
    ax.text(tx, .53, "excess error\nthree to five\nrounds later", fontsize=5.5, va="top", linespacing=1.05)
    for x in (2.27, 5.08):
        ax.plot([x, x], [.06, H - .08], color="#D5D8DC", lw=.6)
    save(fig, "fig1_overview", tight=False, svg=True)


# =========================================================================== Figure 2: trajectories
def curve(ax, e, color, ls="-", lw=1.3, label=None, band=True, zorder=3):
    r = np.array([e["round"][str(t)] for t in range(9)])
    ax.plot(range(9), r[:, 0], color=color, ls=ls, lw=lw, label=label, zorder=zorder)
    if band:
        ax.fill_between(range(9), r[:, 1], r[:, 2], color=color, alpha=.13, lw=0, zorder=zorder - 1)


def window(ax):
    ax.axvspan(1.5, 3.5, color=YEL, alpha=.28, lw=0, zorder=0)
    ax.axvspan(5.5, 8.4, color="#000000", alpha=.045, lw=0, zorder=0)
    ax.axhline(0, color="k", lw=.5, zorder=1)
    ax.set_xticks(range(0, 9, 2))
    ax.set_xlim(-.3, 8.4)
    ax.tick_params(pad=1.5)


def fig_trajectories():
    fig, axes = plt.subplots(1, 5, figsize=(7.05, 1.04), gridspec_kw={"wspace": .10})
    for ax, m in zip(axes[:4], MODELS[:4]):
        window(ax)
        curve(ax, eff(m, "abs", "initial_only", "treated"), RED, label="one-time signal ($\\rho{=}0$)")
        curve(ax, eff(m, "abs", "persistent", "treated"), BLUE, label="recurrent signal ($\\rho{=}1$)")
        curve(ax, eff(m, "abs", "initial_only", "no_relay_treated"), GREY, ls=(0, (4, 2)), label="one-time signal, relay blocked", band=False)
        ax.set_title(m, pad=2)
        ax.set_ylim(-4, 58)
        ax.set_yticks([0, 20, 40])
        ax.set_xlabel("round", labelpad=.5)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("excess error (pp)", labelpad=2)
    axes[0].text(2.5, 56.5, "pertur-\nbation", ha="center", va="top", fontsize=5.3, color="#8a6d00", linespacing=.9)
    axes[0].text(7.0, 56.5, "outcome\nwindow", ha="center", va="top", fontsize=5.3, color="#555", linespacing=.9)
    ax = axes[4]
    window(ax)
    m = "Qwen3-30B"
    curve(ax, eff(m, "nl", "initial_only", "treated"), RED)
    curve(ax, eff(m, "nl", "persistent", "treated"), BLUE)
    curve(ax, eff(m, "nl", "initial_only", "no_relay_treated"), GREY, ls=(0, (4, 2)), band=False)
    ax.set_title("Qwen3-30B (natural lang.)", pad=2, fontsize=6.8)
    ax.set_ylim(-6, 92)
    ax.set_yticks([0, 40, 80])
    ax.set_xlabel("round", labelpad=.5)
    ax.yaxis.tick_right()
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.155), columnspacing=1.6, handlelength=2.0)
    save(fig, "fig2_trajectories")


# =========================================================================== Figure 3: local rule + first contact
SHORT = {"Qwen2.5-7B": "Qwen-7B", "Qwen3-14B": "Qwen-14B", "Qwen3-30B": "Qwen-30B", "GLM-4-9B": "GLM-9B", "Llama-3.1-8B": "Llama-8B"}


def fig_local():
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.42), gridspec_kw={"wspace": .52, "width_ratios": [1, 1]})
    ax = axes[0]
    rec = np.array([[N["rules"][f"{m}|persistent"]["switch_by_opposing"][str(k)][0] * 100 for k in range(5)] for m in MODELS])
    ax.fill_between(range(5), rec.min(0), rec.max(0), color="#D9DDDE", lw=0, zorder=1)
    ax.text(3.95, 30, "recurrent\nsignal", fontsize=5.2, color="#6E7B7C", ha="right", va="center", linespacing=.95)
    ax.plot([0, 1, 2, 3, 4], [0, 0, 50, 100, 100], color="k", ls=":", lw=.9, zorder=2)
    ax.text(1.75, 62, "majority\nrule", fontsize=5.2, color="k", ha="right", va="center", linespacing=.95)
    for m in MODELS:
        sw = N["rules"][f"{m}|initial_only"]["switch_by_opposing"]
        ax.plot(range(5), [sw[str(k)][0] * 100 for k in range(5)], color=MCOL[m], lw=1.1, zorder=4)
    ax.set_xticks(range(5))
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("opposing neighbours (of 4)", labelpad=1)
    ax.set_ylabel("agents that switch (%)", labelpad=1)
    ax.set_title("(a) Local rule", pad=3, loc="left")

    ax = axes[1]
    DARK = "#4D5656"
    for i, m in enumerate(MODELS):
        y = len(MODELS) - 1 - i
        for off, grp, c in ((.16, "abs", DARK), (-.16, "nl", RED)):
            p = N["premium"][f"{m}|{grp}|initial_only|treated"]
            a, b = p["flip_ordinary"] * 100, p["flip_injected"] * 100
            ax.plot([a, b], [y + off, y + off], color=c, lw=1.0, zorder=2, solid_capstyle="round")
            ax.scatter(a, y + off, s=10, facecolor="white", edgecolor=c, lw=.8, zorder=3)
            ax.scatter(b, y + off, s=10, color=c, zorder=4)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([SHORT[m] for m in MODELS[::-1]], fontsize=5.8)
    for lab, m in zip(ax.get_yticklabels(), MODELS[::-1]):
        lab.set_color(MCOL[m])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-.5, 5.3)
    ax.set_xlim(0, 104)
    ax.set_xticks([0, 50, 100])
    ax.grid(axis="x", color="#E3E6E6", lw=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("correct agents that flip (%)", labelpad=1)
    ax.set_title("(b) First contact", pad=3, loc="left")
    ax.text(2, 5.05, "abstract task", fontsize=5.2, color=DARK, ha="left", va="center")
    ax.text(102, 5.05, "authority claim", fontsize=5.2, color=RED, ha="right", va="center")
    save(fig, "fig3_local_rules")


# =========================================================================== Figure 4: plane + trade-off
def fig_plane():
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.42), gridspec_kw={"wspace": .48, "width_ratios": [1.05, 1]})
    ax = axes[0]
    ax.plot([0, 82], [0, 82], color="#999", lw=.6, ls=":", zorder=1)
    spec = [("abs", "initial_only", "treated", RED), ("abs", "init_seed1", "treated", RED), ("abs", "init_seed6", "treated", RED),
            ("abs", "init_fullmem", "treated", RED), ("nl", "initial_only", "treated", RED), ("nl", "initial_only", "treated_neutral", RED),
            ("nl", "initial_only", "treated_authority", RED),
            ("abs", "persistent", "treated", BLUE), ("abs", "persist_seed6", "treated", BLUE), ("nl", "persistent", "treated", BLUE),
            ("abs", "initial_only", "no_relay_treated", GREY), ("abs", "persistent", "no_relay_treated", GREY),
            ("nl", "initial_only", "no_relay_treated", GREY), ("nl", "persistent", "no_relay_treated", GREY)]
    for grp, s, a, c in spec:
        for m in MODELS:
            if m == "Llama-3.1-8B" and grp == "abs" and not s.startswith("persist"):
                continue  # control arm at chance: excess error not informative
            e = eff(m, grp, s, a)
            if e is None:
                continue
            ax.scatter(e["peak3"][0], e["tail"][0], marker="o", s=7, color=c, lw=0, zorder=3 if c == RED else 4, alpha=.85)
    ax.set_xlim(-3, 62)
    ax.set_ylim(-5, 84)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0, 40, 80])
    ax.set_xlabel("shock (pp)", labelpad=1)
    ax.set_ylabel("persistence (pp)", labelpad=1)
    ax.set_title("(a) Shock vs. persistence", pad=3, loc="left")
    ax.text(1, 79, "one-time signal", fontsize=5.4, color=RED, ha="left", va="center")
    ax.text(1, 68, "recurrent signal", fontsize=5.4, color=BLUE, ha="left", va="center")
    ax.text(1, 57, "relay blocked", fontsize=5.4, color=GREY, ha="left", va="center")

    ax = axes[1]
    sch = ["initial_only", "anchor25", "anchor50", "persistent"]
    for m in MODELS[:3]:
        es = [eff(m, "abs", s, "treated") for s in sch]
        x = [e["ctrl_tail"] for e in es]
        y = [e["tail"][0] for e in es]
        ax.plot(x, y, color=MCOL[m], lw=1.0, zorder=2, marker="o", ms=2.2)
        ax.scatter(x[0], y[0], s=13, facecolor="white", edgecolor=MCOL[m], lw=.9, zorder=3)
        ax.scatter(x[-1], y[-1], s=13, color=MCOL[m], zorder=3)
    ax.axhline(0, color="k", lw=.5, zorder=1)
    ax.set_xlim(4, 33)
    ax.set_ylim(-2, 19)
    ax.set_xticks([10, 20, 30])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("error without attack (%)", labelpad=1)
    ax.set_ylabel("persistence (pp)", labelpad=1)
    ax.set_title("(b) Cost of protection", pad=3, loc="left")
    ax.text(15.3, 16.4, "$\\rho{=}0$", fontsize=5.4, color="#444", ha="left", va="center")
    ax.text(30.6, 2.6, "$\\rho{=}1$", fontsize=5.4, color="#444", ha="center", va="center")
    hs = [Line2D([], [], color=MCOL[m], lw=1.0, label=SHORT[m]) for m in MODELS[:3]]
    ax.legend(handles=hs, loc="upper right", bbox_to_anchor=(1.06, 1.03), labelspacing=.12, handlelength=1.1, handletextpad=.3,
              borderpad=0, fontsize=5.2)
    save(fig, "fig4_shock_persistence")


if __name__ == "__main__":
    fig_overview()
    fig_trajectories()
    fig_local()
    fig_plane()
    print("plan A figures written to", OUT)
