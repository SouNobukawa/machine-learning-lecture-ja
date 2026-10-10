"""Generate the figures used in text.tex.

Note: matplotlib mathtext has no \\bm, so figures use \\boldsymbol (same look).

Usage (from class_ML/):
    uv run --with matplotlib --with numpy python make_figures.py
Outputs: figures/*.pdf  (vector graphics, included with \includegraphics)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "Hiragino Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 200,
})
NAVY = "#1F3A5F"
RED = "#C0392B"
GRAY = "#7F8C8D"
GREEN = "#1E8449"
INK_T = "#20262E"


def save(fig, name):
    """Save as PDF (for \includegraphics in text.tex) and PNG (for pptx slides)."""
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"  figures/{name}.pdf + .png")


# ---------------------------------------------------------------- chapter 1
def fig_fit():
    """Underfitting / appropriate / overfitting on the same data."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 11)
    y_true = np.sin(2 * np.pi * x) * 0.8
    y = y_true + rng.normal(0, 0.18, x.size)
    xs = np.linspace(-0.02, 1.02, 300)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
    for ax, (M, title) in zip(axes, [(1, "未学習（$M=1$）"), (3, "適切（$M=3$）"), (10, "過学習（$M=10$）")]):
        c = np.polyfit(x, y, M)
        ax.plot(xs, np.sin(2 * np.pi * xs) * 0.8, color=GRAY, ls="--", lw=1.2, label="真の関係")
        ax.plot(xs, np.polyval(c, xs), color=RED, lw=1.8, label="学習した仮説 $h$")
        ax.scatter(x, y, s=22, color=NAVY, zorder=5, label="訓練データ")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$x$")
        ax.set_ylim(-1.7, 1.7)
    axes[0].set_ylabel("$y$")
    axes[0].legend(loc="upper right", fontsize=7)
    save(fig, "ch1_fit")


def fig_error_curve():
    """Training error decreases, generalization error is U-shaped."""
    M = np.linspace(0.6, 10, 300)
    train = 1.0 / (M ** 1.15) + 0.03
    gen = train + 0.016 * np.abs(M - 1) ** 2.1
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(M, train, color=NAVY, lw=2, label=r"訓練誤差（経験損失 $L_D$）")
    ax.plot(M, gen, color=RED, lw=2, label=r"汎化誤差 $L_{\mathrm{gen}}$")
    m_best = M[np.argmin(gen)]
    ax.axvline(m_best, color=GRAY, ls=":", lw=1.2)
    ax.annotate("最適な複雑さ", xy=(m_best, gen.min()), xytext=(m_best + 0.6, gen.min() + 0.5),
                fontsize=8, arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9))
    ax.text(1.35, 0.13, "未学習", fontsize=9, color=GRAY, ha="center")
    ax.text(8.7, 0.13, "過学習", fontsize=9, color=GRAY, ha="center")
    ax.set_xlabel("モデルの複雑さ（仮説空間の大きさ）")
    ax.set_ylabel("誤差")
    ax.set_ylim(0, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper center", fontsize=8)
    save(fig, "ch1_error_curve")


# ---------------------------------------------------------------- chapter 2
def _arrow(ax, tail, head, color, label=None, lw=1.8, ls="-", label_offset=(0.08, 0.08)):
    ax.annotate("", xy=head, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))
    if label:
        mx, my = (tail[0] + head[0]) / 2, (tail[1] + head[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label, color=color, fontsize=10)


def fig_vec_ops():
    """Vector addition (parallelogram) and scalar multiple."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    x, y = np.array([3, 1]), np.array([1, 2.5])
    ax = axes[0]
    _arrow(ax, (0, 0), x, NAVY, r"$\boldsymbol{x}$", label_offset=(0.05, -0.35))
    _arrow(ax, (0, 0), y, GREEN, r"$\boldsymbol{y}$", label_offset=(-0.45, 0.05))
    _arrow(ax, (0, 0), x + y, RED, r"$\boldsymbol{x}+\boldsymbol{y}$", label_offset=(0.05, 0.12))
    _arrow(ax, x, x + y, GREEN, None, lw=1.0, ls=":")
    _arrow(ax, y, x + y, NAVY, None, lw=1.0, ls=":")
    ax.set_title("和：矢印をつなぐ", fontsize=10)
    ax.set_xlim(-0.6, 4.8); ax.set_ylim(-0.8, 4.2)
    ax = axes[1]
    _arrow(ax, (0, 0), x, NAVY, r"$\boldsymbol{x}$", label_offset=(0.05, -0.4))
    _arrow(ax, (0, 0), 1.6 * x, RED, r"$1.6\,\boldsymbol{x}$", label_offset=(0.05, 0.2))
    _arrow(ax, (0, 0), -0.8 * x, GRAY, r"$-0.8\,\boldsymbol{x}$", label_offset=(-0.3, -0.5))
    ax.set_title("スカラー倍：伸縮（負なら反転）", fontsize=10)
    ax.set_xlim(-3.2, 5.6); ax.set_ylim(-1.6, 2.4)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "ch2_vec_ops")


def fig_projection():
    """Inner product = projection length x ||y||."""
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.array([2.2, 2.6]); y = np.array([4.6, 0.95])
    u = y / np.linalg.norm(y)                 # unit vector along y
    n = np.array([-u[1], u[0]])               # unit normal to y
    p = (x @ u) * u                           # orthogonal projection of x onto y

    _arrow(ax, (0, 0), y, GREEN, None, lw=1.8)
    ax.text(y[0] + 0.12, y[1] - 0.05, r"$\boldsymbol{y}$", color=GREEN, fontsize=11)
    _arrow(ax, (0, 0), x, NAVY, None, lw=1.8)
    ax.text(x[0] - 0.6, x[1] + 0.05, r"$\boldsymbol{x}$", color=NAVY, fontsize=11)
    _arrow(ax, (0, 0), p, RED, None, lw=2.6)  # the projection itself, along y
    ax.plot([x[0], p[0]], [x[1], p[1]], color=GRAY, ls=":", lw=1.3)

    # right-angle marker at the foot of the perpendicular
    m = 0.22
    ax.plot([p[0] - u[0]*m, p[0] - u[0]*m + n[0]*m, p[0] + n[0]*m],
            [p[1] - u[1]*m, p[1] - u[1]*m + n[1]*m, p[1] + n[1]*m], color=GRAY, lw=0.9)

    # angle arc between y and x
    th1 = np.arctan2(y[1], y[0]); th2 = np.arctan2(x[1], x[0])
    arc = np.linspace(th1, th2, 60)
    ax.plot(0.95 * np.cos(arc), 0.95 * np.sin(arc), color="k", lw=0.9)
    ax.text(1.02, 0.42, r"$\theta$", fontsize=12)

    # measurement arrow PARALLEL TO y (offset perpendicular), not along the x-axis
    off = -n * 0.42
    ax.annotate("", xy=p + off, xytext=off,
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.1))
    ax.plot([0, off[0]], [0, off[1]], color=RED, lw=0.7, ls=":")
    ax.plot([p[0], p[0] + off[0]], [p[1], p[1] + off[1]], color=RED, lw=0.7, ls=":")
    mid = p / 2 + off - n * 0.30
    ang = np.degrees(np.arctan2(u[1], u[0]))
    ax.text(mid[0], mid[1], r"$\|\boldsymbol{x}\|\cos\theta$", color=RED, fontsize=10,
            ha="center", va="center", rotation=ang, rotation_mode="anchor")

    ax.set_title(r"$\boldsymbol{x}^\top\boldsymbol{y} = \|\boldsymbol{x}\|\,\|\boldsymbol{y}\|\cos\theta$"
                 "\n（赤は $\\boldsymbol{x}$ の $\\boldsymbol{y}$ 方向への正射影）", fontsize=9)
    ax.axhline(0, color="k", lw=0.5, zorder=0); ax.axvline(0, color="k", lw=0.5, zorder=0)
    ax.set_xlim(-0.9, 5.6); ax.set_ylim(-1.5, 3.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "ch2_projection")


def fig_cossim():
    """Same direction but different length vs different direction."""
    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    a = np.array([3, 4]); b = np.array([6, 8]); c = np.array([4, 3])
    _arrow(ax, (0, 0), b, GRAY, None, lw=2.6)
    ax.text(b[0] - 2.1, b[1] - 0.1, r"$\boldsymbol{b}=2\boldsymbol{a}$", color=GRAY, fontsize=10)
    _arrow(ax, (0, 0), a, NAVY, None, lw=2.0)
    ax.text(a[0] - 1.0, a[1] - 0.15, r"$\boldsymbol{a}$", color=NAVY, fontsize=11)
    _arrow(ax, (0, 0), c, RED, None, lw=2.0)
    ax.text(c[0] + 0.15, c[1] - 0.25, r"$\boldsymbol{c}$", color=RED, fontsize=11)
    arc = np.linspace(np.arctan2(c[1], c[0]), np.arctan2(a[1], a[0]), 60)
    ax.plot(1.7 * np.cos(arc), 1.7 * np.sin(arc), color="k", lw=0.9)
    ax.text(1.55, 1.35, r"$\theta$", fontsize=11)
    ax.set_title("向きが同じなら長さが違っても\nコサイン類似度は $1$", fontsize=9)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-0.7, 7.4); ax.set_ylim(-0.7, 9.0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "ch2_cossim")


def fig_attention():
    """Softmax weights in the numeric example."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    labels = ["猫\n$\\boldsymbol{k}_1$", "魚\n$\\boldsymbol{k}_2$", "数学\n$\\boldsymbol{k}_3$"]
    score = np.array([2.0, 1.0, 0.0])
    alpha = np.exp(score) / np.exp(score).sum()
    ax = axes[0]
    ax.bar(labels, score, color=NAVY, width=0.55)
    ax.set_title(r"スコア $\boldsymbol{q}^\top\boldsymbol{k}_i$", fontsize=10)
    ax.set_ylim(0, 2.5)
    for i, v in enumerate(score):
        ax.text(i, v + 0.08, f"{v:.0f}", ha="center", fontsize=9)
    ax = axes[1]
    ax.bar(labels, alpha, color=RED, width=0.55)
    ax.set_title(r"ソフトマックス後の重み $\alpha_i$", fontsize=10)
    ax.set_ylim(0, 0.85)
    for i, v in enumerate(alpha):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=9)
    axes[1].axhline(0, color="k", lw=0.6)
    save(fig, "ch2_attention")


def fig_neuron():
    """AND neuron: decision boundary and the weight vector as its normal."""
    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    out = np.array([0, 0, 0, 1])
    xs = np.linspace(-0.45, 1.75, 10)
    ax.fill_between(xs, 1.5 - xs, 1.9, color=RED, alpha=0.10)
    ax.plot(xs, 1.5 - xs, color=RED, lw=1.8, label=r"決定境界 $x_1+x_2=1.5$")
    for p, o in zip(pts, out):
        ax.scatter(*p, s=95, marker="o" if o else "x",
                   color=NAVY if o else GRAY, zorder=5, linewidths=2)
        dx, dy = (0.08, 0.06) if p[0] < 0.5 else (0.08, -0.16)
        ax.text(p[0] + dx, p[1] + dy, f"$a={o}$", fontsize=9)
    foot = np.array([0.35, 1.15])           # a point on the boundary, away from the data
    _arrow(ax, foot, foot + np.array([1, 1]) * 0.30, GREEN, None, lw=2.0)
    ax.text(foot[0] - 0.38, foot[1] + 0.42, r"$\boldsymbol{w}=(1,1)^\top$", color=GREEN, fontsize=9)
    ax.text(foot[0] - 0.40, foot[1] + 0.26, "（法線ベクトル）", color=GREEN, fontsize=8)
    ax.set_title("ANDを計算するニューロン", fontsize=10)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_xlim(-0.45, 1.75); ax.set_ylim(-0.45, 1.9)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "ch2_neuron")


# ------------------------------------------------- chapter 2: schematic diagrams
def _box(ax, xy, w, h, text, fc="#FFFFFF", ec=NAVY, fs=9, lw=1.2, tc="black"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fc, edgecolor=ec, lw=lw, zorder=3))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs, zorder=4, color=tc)


def _link(ax, p, q, color=GRAY, lw=1.0, label=None, ls="-", fs=8, shrink=0.16):
    import numpy as _np
    p, q = _np.array(p, float), _np.array(q, float)
    d = q - p
    n = _np.linalg.norm(d)
    if n > 0:
        p2, q2 = p + d / n * shrink, q - d / n * shrink
    else:
        p2, q2 = p, q
    ax.annotate("", xy=q2, xytext=p2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))
    if label:
        m = (p2 + q2) / 2
        ax.text(m[0], m[1] + 0.10, label, ha="center", fontsize=fs, color=color)


def fig_neuron_diagram():
    """One neuron: inputs -> weighted sum (inner product) -> activation -> output."""
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    xs = [2.6, 1.6, 0.4]
    labels = [r"$x_1$", r"$x_2$", r"$x_d$"]
    wlab = [r"$w_1$", r"$w_2$", r"$w_d$"]
    for y, lab, wl in zip(xs, labels, wlab):
        _box(ax, (0.6, y), 0.7, 0.55, lab, fc="#EEF2F7")
        _link(ax, (0.95, y), (2.62, 1.5), label=wl, fs=9)
    ax.text(0.6, 1.02, r"$\vdots$", ha="center", fontsize=13)
    _box(ax, (3.0, 1.5), 1.35, 0.8, r"$z=\boldsymbol{w}^\top\boldsymbol{x}+b$",
         fc="#FDF3D0", ec=RED, fs=10)
    ax.text(3.0, 0.72, "内積（重み付き和）", ha="center", fontsize=8, color=RED)
    _box(ax, (5.0, 1.5), 1.15, 0.8, r"$a=\sigma(z)$", fc="#E3F0E3", ec=GREEN, fs=10)
    ax.text(5.0, 0.72, "活性化関数（非線形）", ha="center", fontsize=8, color=GREEN)
    _link(ax, (3.7, 1.5), (4.4, 1.5), color="black", lw=1.4)
    _link(ax, (5.6, 1.5), (6.45, 1.5), color="black", lw=1.4)
    ax.text(6.6, 1.5, "出力", va="center", fontsize=9)
    _box(ax, (1.75, 0.05), 0.6, 0.45, r"$b$", fc="#EEF2F7", fs=9)
    _link(ax, (2.05, 0.18), (2.62, 1.10))
    ax.text(0.6, 3.1, "入力", ha="center", fontsize=9)
    ax.set_xlim(-0.1, 7.3); ax.set_ylim(-0.45, 3.4)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "ch2_neuron_diagram")


def fig_layer_diagram():
    """A layer is m inner products; stacking them gives the matrix W."""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ys_in = [2.75, 2.0, 1.25, 0.3]
    in_lab = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_d$"]
    ys_out = [2.75, 1.85, 0.45]
    out_lab = [r"$a_1=\sigma(\boldsymbol{w}_1^\top\boldsymbol{x}+b_1)$",
               r"$a_2=\sigma(\boldsymbol{w}_2^\top\boldsymbol{x}+b_2)$",
               r"$a_m=\sigma(\boldsymbol{w}_m^\top\boldsymbol{x}+b_m)$"]
    for y, lab in zip(ys_in, in_lab):
        _box(ax, (0.55, y), 0.68, 0.5, lab, fc="#EEF2F7")
    ax.text(0.55, 0.80, r"$\vdots$", ha="center", fontsize=13)
    for y, lab in zip(ys_out, out_lab):
        _box(ax, (3.8, y), 2.9, 0.62, lab, fc="#FDF3D0", ec=RED, fs=9)
        for yi in ys_in:
            _link(ax, (0.9, yi), (2.35, y), lw=0.55)
    ax.text(3.8, 1.16, r"$\vdots$", ha="center", fontsize=13)
    ax.text(0.55, 3.30, "入力 $\\boldsymbol{x}\\in\\mathbb{R}^d$", ha="center", fontsize=9)
    ax.text(3.8, 3.30, "$m$ 個のニューロン＝$m$ 本の内積", ha="center", fontsize=9, color=RED)
    ax.annotate("", xy=(6.65, 1.6), xytext=(5.45, 1.6),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
    ax.text(6.05, 1.78, "行列で\nまとめる", ha="center", fontsize=8)
    _box(ax, (7.7, 1.6), 1.9, 0.75, r"$\boldsymbol{a}=\sigma(\boldsymbol{W}\boldsymbol{x}+\boldsymbol{b})$",
         fc="#E3F0E3", ec=GREEN, fs=10)
    ax.text(7.7, 0.95, "第3回：行列", ha="center", fontsize=8, color=GREEN)
    ax.set_xlim(-0.1, 8.8); ax.set_ylim(-0.1, 3.6)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "ch2_layer_diagram")


def fig_attention_flow():
    """Data flow of (single-head) attention: q,k,v -> score -> softmax -> weighted sum."""
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ys = [3.05, 2.15, 1.25]
    words = ["猫", "魚", "数学"]
    score = [2.0, 1.0, 0.0]
    alpha = [0.67, 0.24, 0.09]
    _box(ax, (0.75, 0.3), 1.3, 0.55, r"Query $\boldsymbol{q}$", fc="#FDF3D0", ec=RED, fs=10)
    ax.text(0.75, -0.25, "「食べた」", ha="center", fontsize=8, color=RED)
    for y, w, sc, al in zip(ys, words, score, alpha):
        _box(ax, (0.75, y), 1.5, 0.55, f"Key  $\\boldsymbol{{k}}$（{w}）", fc="#EEF2F7", fs=9)
        _link(ax, (1.55, y), (2.75, y), color=NAVY, lw=1.0)
        _box(ax, (3.3, y), 1.15, 0.55, f"$\\boldsymbol{{q}}^\\top\\boldsymbol{{k}}={sc:.0f}$",
             fc="#FFFFFF", ec=NAVY, fs=9)
        _link(ax, (1.2, 0.55), (2.78, y - 0.22), color=RED, lw=0.8, ls=":")
        _link(ax, (3.9, y), (4.75, y), color=GRAY, lw=1.0)
        _box(ax, (5.35, y), 1.25, 0.55, f"$\\alpha={al:.2f}$", fc="#E3F0E3", ec=GREEN, fs=9)
        _link(ax, (6.0, y), (6.93, 2.35), color=GREEN, lw=1.0)
    ax.text(3.3, 3.72, "スコア（内積）", ha="center", fontsize=9, color=NAVY)
    ax.text(5.35, 3.72, "重み（softmax）", ha="center", fontsize=9, color=GREEN)
    ax.annotate("", xy=(5.35, 0.62), xytext=(5.35, 0.98),
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, ls=":"))
    ax.text(5.35, 0.42, r"$\sum_i \alpha_i = 1$", ha="center", fontsize=9, color=GREEN)
    _box(ax, (7.6, 2.35), 1.35, 0.8, r"$\sum_i \alpha_i \boldsymbol{v}_i$", fc="#FDF3D0", ec=RED, fs=11)
    ax.text(7.6, 1.60, "Value の重み付き和", ha="center", fontsize=8, color=RED)
    ax.text(7.6, 1.30, "（出力）", ha="center", fontsize=8, color=RED)
    ax.set_xlim(-0.2, 8.6); ax.set_ylim(-0.5, 3.95)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "ch2_attention_flow")


def fig_embedding():
    """One-hot (orthogonal, meaningless) vs embedding (directions carry meaning)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    words = ["猫", "犬", "数学"]
    for i, w in enumerate(words):
        for j in range(3):
            ax.add_patch(plt.Rectangle((j, -i - 0.4), 0.9, 0.8,
                                       facecolor=NAVY if i == j else "#FFFFFF",
                                       edgecolor=GRAY, lw=0.8))
            ax.text(j + 0.45, -i, "1" if i == j else "0", ha="center", va="center",
                    fontsize=9, color="white" if i == j else GRAY)
        ax.text(-0.35, -i, w, ha="right", va="center", fontsize=10)
    ax.text(1.35, 0.95, "ワンホット（$V$ 次元）", ha="center", fontsize=10)
    ax.text(1.35, -2.95, r"どの2語も直交：$\boldsymbol{o}_i^\top\boldsymbol{o}_j=0$",
            ha="center", fontsize=9, color=RED)
    ax.text(1.35, -3.45, "意味の近さを測れない", ha="center", fontsize=9, color=RED)
    ax.set_xlim(-1.3, 3.2); ax.set_ylim(-3.8, 1.3)
    ax.set_aspect("equal"); ax.axis("off")

    ax = axes[1]
    vecs = {"猫": (2.6, 2.3), "犬": (2.9, 1.9), "数学": (-1.3, 2.6)}
    colors = {"猫": NAVY, "犬": NAVY, "数学": GRAY}
    for w, v in vecs.items():
        _arrow(ax, (0, 0), v, colors[w], None, lw=2.0)
        ax.text(v[0] + (0.12 if v[0] > 0 else -0.55), v[1] + 0.12, w, fontsize=10, color=colors[w])
    arc = np.linspace(np.arctan2(1.9, 2.9), np.arctan2(2.3, 2.6), 40)
    ax.plot(1.5 * np.cos(arc), 1.5 * np.sin(arc), color=RED, lw=1.2)
    ax.text(1.85, 1.35, r"小", fontsize=9, color=RED)
    arc2 = np.linspace(np.arctan2(2.3, 2.6), np.arctan2(2.6, -1.3), 40)
    ax.plot(1.05 * np.cos(arc2), 1.05 * np.sin(arc2), color=GRAY, lw=1.0, ls=":")
    ax.text(0.15, 1.35, r"大", fontsize=9, color=GRAY)
    ax.set_title("埋め込み（$d$ 次元、$d \\ll V$）", fontsize=10)
    ax.text(0.7, -0.75, "向きの違いが意味の違いを表す\n→ コサイン類似度で測れる",
            ha="center", fontsize=9, color=NAVY)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-2.4, 4.0); ax.set_ylim(-1.5, 3.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "ch2_embedding")


def fig_span():
    """Schematic: Ax lives on the plane spanned by the columns of A."""
    # screen directions for the two columns (a readable projection, not to scale)
    u = np.array([1.00, 0.16])      # c1
    v = np.array([0.34, 0.92])      # c2

    def P(a, b):
        return a * u + b * v

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    corners = [P(-0.9, -1.3), P(2.9, -1.3), P(2.9, 1.5), P(-0.9, 1.5)]
    ax.add_patch(plt.Polygon(corners, closed=True, facecolor="#DCE6F1",
                             edgecolor=NAVY, lw=1.1, alpha=0.8, zorder=0))

    def arrow(p, color, label, off=(0, 0), lw=2.2, ls="-"):
        ax.annotate("", xy=tuple(p), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                    shrinkA=0, shrinkB=0))
        if label:
            ax.text(p[0] + off[0], p[1] + off[1], label, color=color, fontsize=12)

    target = P(2, -1)
    # parallelogram construction: go 2 c1, then -1 c2
    ax.annotate("", xy=tuple(target), xytext=tuple(P(2, 0)),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5, ls="--"))
    ax.text(*(P(2, -0.5) + np.array([0.10, 0.04])), r"$-\boldsymbol{c}_2$",
            color=GREEN, fontsize=11)
    arrow(P(2, 0), GRAY, r"$2\boldsymbol{c}_1$", (0.02, 0.14), lw=1.5, ls="--")
    arrow(u, NAVY, r"$\boldsymbol{c}_1$", (-0.10, 0.16))
    arrow(v, GREEN, r"$\boldsymbol{c}_2$", (-0.34, 0.06))
    arrow(target, RED, None, lw=3.0)
    ax.text(target[0] + 0.10, target[1] - 0.06,
            r"$\boldsymbol{A}\boldsymbol{x} = 2\boldsymbol{c}_1 - \boldsymbol{c}_2$",
            color=RED, fontsize=12)

    ax.scatter(0, 0, s=24, color="k", zorder=6)
    ax.text(-0.16, -0.22, r"$\boldsymbol{0}$", fontsize=11)
    ax.text(0.62, 1.18, r"$\boldsymbol{c}_1,\ \boldsymbol{c}_2$ が張る平面",
            color=NAVY, fontsize=11)
    ax.set_title(r"$\boldsymbol{A}\boldsymbol{x}$ は列ベクトルの張る平面の上にしか現れない",
                 fontsize=11)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlim(-1.3, 3.6); ax.set_ylim(-1.5, 1.9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "ch3_span")


def fig_independence():
    """Linearly independent columns span a plane; dependent ones only a line."""
    u = np.array([1.00, 0.16])
    v = np.array([0.34, 0.92])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))

    def arrow(ax, p, color, label, off=(0, 0), lw=2.4, ls="-"):
        ax.annotate("", xy=tuple(p), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                    shrinkA=0, shrinkB=0))
        if label:
            ax.text(p[0] + off[0], p[1] + off[1], label, color=color, fontsize=12)

    # --- (a) independent: a plane
    ax = axes[0]
    corners = [(-0.9, -1.1), (2.5, -1.1), (2.5, 1.4), (-0.9, 1.4)]
    ax.add_patch(plt.Polygon([a * u + b * v for a, b in corners], closed=True,
                             facecolor="#DCE6F1", edgecolor=NAVY, lw=1.1, alpha=0.8, zorder=0))
    arrow(ax, u, NAVY, r"$\boldsymbol{c}_1$", (-0.06, 0.16))
    arrow(ax, v, GREEN, r"$\boldsymbol{c}_2$", (-0.34, 0.06))
    ax.text(0.30, 1.16, "張られるのは\n平面（2次元）", color=NAVY, fontsize=10)
    ax.set_title("線形独立：どちらも他方のスカラー倍でない", fontsize=10)
    ax.set_xlim(-1.3, 3.2); ax.set_ylim(-1.3, 1.9)

    # --- (b) dependent: only a line
    ax = axes[1]
    d = np.array([1.00, 0.34])
    ax.plot([-1.5 * d[0], 3.1 * d[0]], [-1.5 * d[1], 3.1 * d[1]],
            color=NAVY, lw=8, alpha=0.22, solid_capstyle="round", zorder=0)
    arrow(ax, 2.1 * d, GREEN, r"$\boldsymbol{c}_2 = 2\boldsymbol{c}_1$", (0.08, -0.22), lw=2.4)
    arrow(ax, 1.05 * d, NAVY, r"$\boldsymbol{c}_1$", (-0.10, 0.20))
    ax.text(-1.25, 1.12, "張られるのは\n直線（1次元）", color=NAVY, fontsize=10)
    ax.set_title("線形従属：一方が他方のスカラー倍", fontsize=10)
    ax.set_xlim(-1.7, 3.6); ax.set_ylim(-1.3, 1.9)

    for ax in axes:
        ax.scatter(0, 0, s=22, color="k", zorder=6)
        ax.text(-0.16, -0.26, r"$\boldsymbol{0}$", fontsize=11)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    save(fig, "ch3_independence")


def fig_linearmap():
    """A linear map sends grids to grids and parallelograms to parallelograms."""
    A = np.array([[1.05, 0.55], [0.30, 0.95]])
    x = np.array([1.5, 0.35])
    y = np.array([0.45, 1.35])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.9))

    def draw(ax, M, title, lab, offs, xlim, ylim):
        # a generous grid of the domain, pushed through M
        g0, g1 = -1.2, 4.0
        for t in np.arange(g0, g1 + 0.01, 0.5):
            for a, b in (((t, g0), (t, g1)), ((g0, t), (g1, t))):
                p0, p1 = M @ np.array(a), M @ np.array(b)
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GRAY, lw=0.6, alpha=0.5, zorder=0)
        o = np.zeros(2)
        px, py, ps = M @ x, M @ y, M @ (x + y)
        ax.add_patch(plt.Polygon([o, px, ps, py], closed=True,
                                 facecolor="#CFDDEE", edgecolor="none", alpha=0.85, zorder=1))
        for p0, p1 in ((px, ps), (py, ps)):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GRAY, ls=":", lw=1.1, zorder=2)
        for v, c, t, off in ((px, NAVY, lab[0], offs[0]),
                             (py, GREEN, lab[1], offs[1]),
                             (ps, RED, lab[2], offs[2])):
            ax.annotate("", xy=tuple(v), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2, shrinkA=0, shrinkB=0))
            ax.text(v[0] + off[0], v[1] + off[1], t, color=c, fontsize=12, zorder=5)
        ax.scatter(0, 0, s=22, color="k", zorder=6)
        ax.text(-0.20, -0.34, r"$\boldsymbol{0}$", fontsize=11, zorder=6)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    draw(axes[0], np.eye(2), r"定義域 $\mathbb{R}^n$",
         [r"$\boldsymbol{x}$", r"$\boldsymbol{y}$", r"$\boldsymbol{x}+\boldsymbol{y}$"],
         [(0.08, -0.26), (-0.34, 0.10), (0.08, 0.10)], (-0.8, 3.2), (-0.8, 2.9))
    draw(axes[1], A, r"値域 $\mathbb{R}^m$（$f_{\boldsymbol{A}}$ による像）",
         [r"$\boldsymbol{A}\boldsymbol{x}$", r"$\boldsymbol{A}\boldsymbol{y}$",
          r"$\boldsymbol{A}(\boldsymbol{x}+\boldsymbol{y})$"],
         [(0.10, -0.28), (-0.52, 0.14), (-1.05, 0.26)], (-0.8, 3.2), (-0.8, 2.9))

    fig.text(0.498, 0.50, r"$f_{\boldsymbol{A}}$", ha="center", fontsize=15, color=INK_T)
    fig.text(0.498, 0.42, r"$\longrightarrow$", ha="center", fontsize=22, color=INK_T)
    save(fig, "ch3_linearmap")


def fig_matmul():
    """(AB)_ij is the inner product of row i of A and column j of B."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    cw, ch = 0.5, 0.5          # cell size
    m, n, p = 3, 4, 3          # A is m x n, B is n x p, AB is m x p
    hi_i, hi_j = 1, 2          # the highlighted row / column (0-based)

    def grid(x0, y0, rows, cols, label, shade=None):
        for r in range(rows):
            for c in range(cols):
                fc = "#FFFFFF"
                if shade == "row" and r == hi_i:
                    fc = "#FDF3D0"
                if shade == "col" and c == hi_j:
                    fc = "#E3F0E3"
                if shade == "cell" and r == hi_i and c == hi_j:
                    fc = "#F7D6D2"
                ax.add_patch(plt.Rectangle((x0 + c * cw, y0 - (r + 1) * ch), cw, ch,
                                           facecolor=fc, edgecolor=GRAY, lw=0.8))
        ax.text(x0 + cols * cw / 2, y0 + 0.16, label, ha="center", fontsize=13)
        return x0 + cols * cw

    xa = 0.2
    xa_end = grid(xa, 2.0, m, n, r"$\boldsymbol{A}$  ($m \times n$)", shade="row")
    ax.text(xa_end + 0.18, 2.0 - (m * ch) / 2, r"$\times$", fontsize=16, va="center")
    xb = xa_end + 0.45
    xb_end = grid(xb, 2.0, n, p, r"$\boldsymbol{B}$  ($n \times p$)", shade="col")
    ax.text(xb_end + 0.18, 2.0 - (m * ch) / 2, r"$=$", fontsize=16, va="center")
    xc = xb_end + 0.45
    xc_end = grid(xc, 2.0, m, p, r"$\boldsymbol{A}\boldsymbol{B}$  ($m \times p$)", shade="cell")

    # annotate the highlighted row, column and the resulting entry
    ax.annotate("", xy=(xc + (hi_j + 0.5) * cw, 2.0 - (hi_i + 0.5) * ch),
                xytext=(xa + n * cw * 0.5, 2.0 - (hi_i + 0.5) * ch),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4,
                                connectionstyle="arc3,rad=-0.28"))
    ax.text(xa - 0.06, 2.0 - (hi_i + 0.5) * ch, r"第 $i$ 行", ha="right", va="center",
            fontsize=11, color="#B8860B")
    ax.text(xb + (hi_j + 0.5) * cw, 2.0 - n * ch - 0.30, r"第 $j$ 列", ha="center",
            fontsize=11, color=GREEN)
    ax.text(xc_end + 0.12, 2.0 - (hi_i + 0.5) * ch, r"$(\boldsymbol{A}\boldsymbol{B})_{ij}$",
            va="center", fontsize=12, color=RED)

    ax.text((xa + xc_end) / 2, -0.82,
            r"$(\boldsymbol{A}\boldsymbol{B})_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$"
            "　（第 $i$ 行と第 $j$ 列の内積）",
            ha="center", fontsize=12)
    ax.set_xlim(-0.6, xc_end + 1.2); ax.set_ylim(-1.15, 2.5)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "ch3_matmul")


# ---------------------------------------------------------------- chapter 3
def fig_colspace_sweep():
    """Varying w moves yhat only inside Col(X): a low-dimensional slice of R^N."""
    fig, ax = plt.subplots(figsize=(5.8, 3.9))

    # the ambient space R^N
    ax.add_patch(plt.Rectangle((-3.2, -2.0), 7.4, 4.9, facecolor="#FAFAF6",
                               edgecolor=GRAY, lw=1.0, ls="--", zorder=0))
    ax.text(-3.05, 2.60, r"$\mathbb{R}^{N}$（$N$ 次元、$N \gg d$）", color=GRAY, fontsize=10)

    # the column space, drawn as a plane
    P = np.array([[-2.5, -1.15], [2.6, -1.55], [3.4, 0.75], [-1.7, 1.15]])
    ax.add_patch(plt.Polygon(P, closed=True, facecolor="#DCE6F1", edgecolor=NAVY,
                             lw=1.2, alpha=0.9, zorder=1))
    ax.text(-0.80, -1.24, "列ベクトルが張る空間", color=NAVY, fontsize=11, zorder=5)

    u = np.array([1.08, -0.12])     # screen direction of c_0
    v = np.array([0.42, 0.62])      # screen direction of c_1

    def arrow(p, color, lw=2.2, ls="-"):
        ax.annotate("", xy=tuple(p), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                    shrinkA=0, shrinkB=0))

    arrow(u, NAVY); ax.text(u[0] + 0.06, u[1] + 0.10, r"$\boldsymbol{c}_0$", color=NAVY, fontsize=12)
    arrow(v, GREEN); ax.text(v[0] - 0.46, v[1] + 0.04, r"$\boldsymbol{c}_1$", color=GREEN, fontsize=12)

    # a few predictions for different w, all on the plane
    for (a, b), lab, off in (((1.9, -0.8), r"$\boldsymbol{X}\boldsymbol{w}^{(1)}$", (0.12, -0.32)),
                             ((0.7, 1.2), r"$\boldsymbol{X}\boldsymbol{w}^{(2)}$", (-0.20, 0.22)),
                             ((-1.5, 0.5), r"$\boldsymbol{X}\boldsymbol{w}^{(3)}$", (-0.62, 0.24))):
        p = a * u + b * v
        arrow(p, RED, lw=1.6, ls=":")
        ax.scatter(*p, s=48, color=RED, zorder=6)
        ax.text(p[0] + off[0], p[1] + off[1], lab, color=RED, fontsize=10, zorder=6)

    ax.scatter(0, 0, s=24, color="k", zorder=7)
    ax.text(-0.18, -0.34, r"$\boldsymbol{0}$", fontsize=11, zorder=7)
    ax.text(-3.05, -1.80,
            r"$\hat{\boldsymbol{y}} = \boldsymbol{X}\boldsymbol{w} = w_0\boldsymbol{c}_0 + w_1\boldsymbol{c}_1$", fontsize=11)
    ax.set_title(r"$\boldsymbol{w}$ をどう選んでも $\hat{\boldsymbol{y}}$ は列ベクトルが張る空間から出られない",
                 fontsize=11)
    ax.set_xlim(-3.5, 4.6); ax.set_ylim(-2.3, 3.1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "ch3_colspace_sweep")


def fig_colspace():
    """y outside the column space; least squares = orthogonal projection."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    # plane drawn as a parallelogram (schematic 3D)
    P = np.array([[-2.1, -0.75], [2.1, -1.35], [3.0, 0.75], [-1.2, 1.35]])
    ax.add_patch(plt.Polygon(P, closed=True, facecolor="#DCE6F1", edgecolor=NAVY, lw=1.2, alpha=0.85))
    ax.text(2.25, -1.15, r"$\mathrm{Col}(\boldsymbol{X})$", color=NAVY, fontsize=11)
    o = np.array([0.0, 0.0])
    yhat = np.array([1.25, 0.12])
    y = yhat + np.array([0.28, 2.05])
    _arrow(ax, o, y, RED, r"$\boldsymbol{y}$", label_offset=(0.1, 0.05))
    _arrow(ax, o, yhat, NAVY, r"$\hat{\boldsymbol{y}}=\boldsymbol{X}\boldsymbol{w}^*$", label_offset=(-0.35, -0.55))
    ax.annotate("", xy=y, xytext=yhat,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8))
    ax.text(yhat[0] + 0.2, (y[1] + yhat[1]) / 2, r"$\boldsymbol{y}-\hat{\boldsymbol{y}}$" + "\n（誤差）",
            color=GREEN, fontsize=9)
    # right-angle marker
    d1 = (P[1] - P[0]); d1 = d1 / np.linalg.norm(d1) * 0.26
    d2 = (y - yhat); d2 = d2 / np.linalg.norm(d2) * 0.26
    ax.plot([yhat[0] + d1[0], yhat[0] + d1[0] + d2[0], yhat[0] + d2[0]],
            [yhat[1] + d1[1], yhat[1] + d1[1] + d2[1], yhat[1] + d2[1]], color="k", lw=0.9)
    ax.scatter(*o, s=18, color="k", zorder=6)
    ax.text(-0.3, -0.28, r"$\boldsymbol{0}$", fontsize=9)
    ax.set_title(r"最小二乗法＝$\boldsymbol{y}$ の列空間への正射影", fontsize=10)
    ax.set_xlim(-2.6, 4.0); ax.set_ylim(-1.8, 2.9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "ch3_colspace")


def fig_flat_valley():
    """L_D has an almost flat valley along v, so noise slides the optimum far."""
    Xs = np.array([[22., 1.], [38., 2.], [61., 3.], [79., 4.]])   # columns c1, c2
    w_true = np.array([0.25, 5.0])
    y = Xs @ w_true
    G = Xs.T @ Xs

    w1 = np.linspace(-0.35, 1.05, 500)
    w2 = np.linspace(-7.0, 17.0, 500)
    W1, W2 = np.meshgrid(w1, w2)
    L = np.zeros_like(W1)
    for k in range(4):
        L += (y[k] - Xs[k, 0] * W1 - Xs[k, 1] * W2) ** 2
    L /= 4.0

    fill = [0.0, 0.2, 1.0, 5.0, 25.0, 120.0, 600.0, L.max()]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    def backdrop(ax):
        ax.contourf(W1, W2, L, levels=fill, cmap="Blues", alpha=0.55)
        ax.contour(W1, W2, L, levels=fill[1:-1], colors="white", linewidths=0.7)
        ax.set_xlabel(r"$w_1$（広さの係数）", fontsize=15)
        ax.set_ylabel(r"$w_2$（部屋数の係数）", fontsize=15)
        ax.set_xlim(w1[0], w1[-1]); ax.set_ylim(w2[0], w2[-1])
        ax.grid(False)

    # ---- left: the valley itself
    ax = axes[0]
    backdrop(ax)
    t = np.linspace(-0.9, 0.9, 2)
    ax.plot(w_true[0] + t, w_true[1] - 20 * t, ls="--", color=RED, lw=1.4, zorder=3)
    ax.text(0.60, -3.1, r"谷の底（$\boldsymbol{v}$ の方向）", color=RED, fontsize=14.2)

    ax.scatter(*w_true, marker="*", s=180, color="k", zorder=6)
    ax.text(w_true[0] - 0.19, w_true[1] - 0.3, r"$\boldsymbol{w}^*$", fontsize=16.5, zorder=6)
    for w, lab, off in (((0.5, 0.0), r"$\boldsymbol{w}^{(\mathrm{A})}$", (0.05, 0.6)),
                        ((0.0, 10.0), r"$\boldsymbol{w}^{(\mathrm{B})}$", (0.05, 0.6))):
        ax.scatter(*w, s=60, color=GREEN, zorder=6)
        ax.text(w[0] + off[0], w[1] + off[1], lab, color=GREEN, fontsize=15.8, zorder=6)

    ax.annotate("", xy=(0.66, 11.0), xytext=(0.28, 5.4),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.7))
    ax.text(0.47, 12.2, "この向きには\n急に増える", color="#333333", fontsize=14.2)
    ax.text(0.03, 0.03, "色が濃いほど $L_D$ が大きい", transform=ax.transAxes,
            fontsize=12.8, color="#333333")
    ax.set_title(r"$L_D$ の谷は $\boldsymbol{v}$ の方向に極端に細長い", fontsize=16.5)

    # ---- right: noise slides the optimum along the valley
    ax = axes[1]
    backdrop(ax)
    rng = np.random.default_rng(3)
    found = []
    while len(found) < 2:
        d = rng.normal(0.0, 0.3, 4)
        ws = np.linalg.solve(G, Xs.T @ (y + d))
        if w1[0] + 0.12 < ws[0] < w1[-1] - 0.12 and w2[0] + 1.5 < ws[1] < w2[-1] - 1.5:
            found.append(ws)
    (a, b) = found
    t2 = np.linspace(-0.9, 0.9, 2)
    ax.plot(w_true[0] + t2, w_true[1] - 20 * t2, ls="--", color=RED, lw=1.0,
            alpha=0.45, zorder=2)
    ax.annotate("", xy=tuple(b), xytext=tuple(a),
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.6))
    ax.scatter(*a, marker="X", s=100, color=RED, zorder=6)
    ax.text(a[0] - 0.05, a[1] - 1.9, f"データ1：$({a[0]:.2f},\\ {a[1]:.1f})$",
            color=RED, fontsize=13.5, ha="right", zorder=6)
    ax.scatter(*b, marker="X", s=100, color=RED, zorder=6)
    ax.text(b[0] + 0.05, b[1] + 1.1, f"データ2：$({b[0]:.2f},\\ {b[1]:.1f})$",
            color=RED, fontsize=13.5, ha="left", zorder=6)
    ax.set_title("家賃に $0.3$ 万円程度の誤差を加えただけで\n最適解が谷に沿って滑る", fontsize=16.5)

    save(fig, "ch3_flat_valley")


def fig_eigen():
    """Eigenvectors keep their direction; the eigenvalue is the stretch factor."""
    A = np.array([[3.0, 1.0], [1.0, 3.0]])          # eigenvalues 4 and 2
    v1 = np.array([1.0, 1.0]) / np.sqrt(2)          # lambda = 4
    v2 = np.array([1.0, -1.0]) / np.sqrt(2)         # lambda = 2

    # drawn at final print size (text width is about 6.3 in)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3))

    def frame(ax, lim):
        ax.axhline(0, color=GRAY, lw=0.7); ax.axvline(0, color=GRAY, lw=0.7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
        ax.set_xticks([]); ax.set_yticks([])

    def arrow(ax, p, color, lw=1.8, ls="-"):
        ax.annotate("", xy=tuple(p), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                    shrinkA=0, shrinkB=0))

    # ---- left: eigen directions vs a general direction
    ax = axes[0]
    frame(ax, 3.7)
    for v, col in ((v1, RED), (v2, GREEN)):
        ax.plot([-3.5 * v[0], 3.5 * v[0]], [-3.5 * v[1], 3.5 * v[1]],
                ls=":", color=col, lw=0.8, alpha=0.7)
        arrow(ax, A @ v, col, lw=2.0)
        arrow(ax, v, col, lw=1.1)
        ax.scatter(*v, s=12, color=col, zorder=5)
    ax.text(0.92, 0.42, r"$\boldsymbol{v}_1$", color=RED, fontsize=9)
    ax.text(0.95, 2.95, r"$\boldsymbol{A}\boldsymbol{v}_1 = 4\boldsymbol{v}_1$",
            color=RED, fontsize=9)
    ax.text(0.14, -1.05, r"$\boldsymbol{v}_2$", color=GREEN, fontsize=9)
    ax.text(0.55, -2.25, r"$\boldsymbol{A}\boldsymbol{v}_2 = 2\boldsymbol{v}_2$",
            color=GREEN, fontsize=9)

    x = np.array([1.0, 0.0])
    arrow(ax, x, NAVY, lw=1.1)
    arrow(ax, A @ x, NAVY, lw=2.0)
    ax.text(0.70, -0.45, r"$\boldsymbol{x}$", color=NAVY, fontsize=9)
    ax.text(2.45, 1.25, r"$\boldsymbol{A}\boldsymbol{x}$", color=NAVY, fontsize=9)
    ax.text(0.02, 0.03, "固有ベクトル（赤・緑）は向きが変わらない\nそれ以外（紺）は向きも変わる",
            transform=ax.transAxes, fontsize=7, color="#333333")
    ax.set_title("固有ベクトル＝向きの変わらない方向", fontsize=9)

    # ---- right: unit circle -> ellipse with axes along the eigenvectors
    ax = axes[1]
    frame(ax, 4.7)
    th = np.linspace(0, 2 * np.pi, 400)
    C = np.vstack([np.cos(th), np.sin(th)])
    E = A @ C
    ax.plot(C[0], C[1], color=GRAY, lw=1.0, ls="--")
    ax.plot(E[0], E[1], color=NAVY, lw=1.4)
    arrow(ax, 4 * v1, RED, lw=1.9)
    arrow(ax, 2 * v2, GREEN, lw=1.9)
    ax.text(1.35, 3.25, r"$\lambda_1 = 4$ 倍", color=RED, fontsize=9)
    ax.text(1.55, -2.15, r"$\lambda_2 = 2$ 倍", color=GREEN, fontsize=9)
    ax.text(-1.55, 1.05, "単位円", color=GRAY, fontsize=8)
    ax.text(0.03, 0.92, "像は楕円", transform=ax.transAxes, color=NAVY, fontsize=8.5)
    ax.text(0.02, 0.03, "楕円の軸の向きが固有ベクトル、\n軸の長さの比が固有値の比",
            transform=ax.transAxes, fontsize=7, color="#333333")
    ax.set_title("固有値＝その方向の拡大率", fontsize=9)

    save(fig, "ch3_eigen")


def fig_eigen_ml():
    """Three places where eigenvalues show up in machine learning."""
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5))

    # ---- 1. PCA: the principal axes are eigenvectors of the covariance
    ax = axes[0]
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (220, 2)) * np.array([1.5, 0.42])
    th = np.deg2rad(32.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = base @ R.T
    ax.scatter(P[:, 0], P[:, 1], s=3.5, color=NAVY, alpha=0.40)
    e1 = R @ np.array([1.0, 0.0])
    e2 = R @ np.array([0.0, 1.0])
    for v, L, col in ((e1, 2.7, RED), (e2, 0.95, GREEN)):
        ax.annotate("", xy=tuple(L * v), xytext=tuple(-L * v),
                    arrowprops=dict(arrowstyle="<|-|>", color=col, lw=1.6))
    ax.text(-0.10, -2.35, r"第1主成分（$\lambda_{\max}$）", color=RED, fontsize=7)
    ax.text(-3.80, 2.05, r"第2主成分（$\lambda_{\min}$）", color=GREEN, fontsize=7)
    ax.set_xlim(-3.9, 3.9); ax.set_ylim(-2.7, 2.7)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.25)
    ax.set_title("主成分分析\n分散最大＝$\\lambda_{\\max}$ の固有ベクトル", fontsize=7.5)

    # ---- 2. gradient descent zigzags when the eigenvalues differ a lot
    ax = axes[1]
    lam = np.array([1.0, 9.0])
    g1, g2 = np.meshgrid(np.linspace(-5.4, 5.4, 300), np.linspace(-2.7, 2.7, 300))
    F = 0.5 * (lam[0] * g1 ** 2 + lam[1] * g2 ** 2)
    ax.contour(g1, g2, F, levels=[0.4, 1.5, 4.0, 8.0, 14.0], colors=NAVY,
               linewidths=0.6, alpha=0.55)
    w = np.array([4.6, 1.7]); path = [w.copy()]
    alpha = 2.0 / (lam[0] + lam[1])
    for _ in range(15):
        w = w - alpha * (lam * w)
        path.append(w.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "-o", color=RED, lw=1.0, ms=2.2, zorder=4)
    ax.scatter(*path[0], s=22, color=RED, zorder=5)
    ax.text(path[0, 0] - 0.2, path[0, 1] + 0.30, "スタート", color=RED,
            fontsize=7, ha="right")
    ax.scatter(0, 0, marker="*", s=70, color="k", zorder=6)
    ax.text(-0.35, 0.25, "最小点", fontsize=7, ha="right")
    ax.text(0.02, 0.03, r"固有値の比 $\lambda_{\max}/\lambda_{\min}$ が" + "\n大きいほどジグザグして遅い",
            transform=ax.transAxes, fontsize=7, color="#333333")
    ax.set_xlim(-5.4, 5.4); ax.set_ylim(-2.7, 2.7)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("勾配降下法の収束（第5回）\n曲がり具合＝ヘッセ行列の固有値", fontsize=7.5)

    # ---- 3. the smallest eigenvalue controls how flat the valley is
    ax = axes[2]
    t = np.linspace(0, 2 * np.pi, 400)
    # level set { w : w^T (X^T X) w = const }; semi-axis = 1/sqrt(lambda)
    ax.plot(0.95 * np.cos(t), 0.80 * np.sin(t), color=GREEN, lw=1.4)
    ax.plot(0.30 * np.cos(t), 2.30 * np.sin(t), color=RED, lw=1.4)
    ax.text(-3.30, 1.45, r"$\lambda_{\min}$ が大きい" + "\n（丸い＝安定）",
            color=GREEN, fontsize=7)
    ax.text(0.45, 1.75, r"$\lambda_{\min} \approx 0$" + "\n（細長い＝不安定）",
            color=RED, fontsize=7)
    ax.text(0.02, 0.03, r"細長い向き＝$L_D$ がほとんど" + "\n増えない向き（図18 の平らな谷）",
            transform=ax.transAxes, fontsize=7, color="#333333")
    ax.axhline(0, color=GRAY, lw=0.7); ax.axvline(0, color=GRAY, lw=0.7)
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.7, 2.7)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.2)
    ax.set_title("数値的安定性\n" + r"$\boldsymbol{X}^\top\boldsymbol{X}$ の等高線の形", fontsize=7.5)

    save(fig, "ch3_eigen_ml")

# ---------------------------------------------------------------- chapter 4
def fig_param_to_distance():
    """The same choice seen twice: pick a w on the left, land on a point on the right."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0),
                             gridspec_kw={"width_ratios": [1.0, 1.45]})
    cols = [NAVY, GREEN, "#B5651D"]
    labs = [r"$\boldsymbol{w}^{(1)}$", r"$\boldsymbol{w}^{(2)}$", r"$\boldsymbol{w}^{(3)}$"]

    # ---- left: the parameter space, nothing special about any point
    ax = axes[0]
    ax.add_patch(plt.Rectangle((-1.5, -1.3), 3.0, 2.6, facecolor="#FAFAF6",
                               edgecolor=GRAY, lw=1.0, ls="--"))
    ax.text(-1.42, 1.10, r"$\mathbb{R}^{d+1}$（パラメータの空間）", color=GRAY, fontsize=8)
    W = [(-0.75, 0.52), (0.62, 0.70), (0.18, -0.72)]
    for (x, y), c, lab in zip(W, cols, labs):
        ax.scatter(x, y, s=48, color=c, zorder=5)
        ax.text(x + 0.10, y + 0.16, lab, color=c, fontsize=9)
    ax.text(0.0, -1.62, "どの $\\boldsymbol{w}$ を選ぶか", ha="center", fontsize=9, color=INK_T)
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-2.0, 1.6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # ---- right: the output space, where the choice becomes a distance
    ax = axes[1]
    P = np.array([[-2.0, -0.95], [2.0, -1.45], [2.8, 0.45], [-1.2, 0.95]])
    ax.add_patch(plt.Polygon(P, closed=True, facecolor="#DCE6F1", edgecolor=NAVY,
                             lw=1.0, alpha=0.85, zorder=1))
    ax.text(1.42, -1.14, r"$\mathrm{Col}(\boldsymbol{X})$", color=NAVY, fontsize=8.5)
    ax.text(-2.5, 1.95, r"$\mathbb{R}^{N}$（予測の空間）", color=GRAY, fontsize=8)

    y = np.array([0.55, 1.75])
    V = [np.array([-1.05, -0.28]), np.array([1.35, -0.42]), np.array([0.30, -0.02])]
    for v, c in zip(V, cols):
        ax.plot([v[0], y[0]], [v[1], y[1]], color=c, lw=1.1, ls=":", zorder=3)
        ax.scatter(*v, s=48, color=c, zorder=5)
    ax.text(V[0][0] - 0.52, V[0][1] - 0.46, r"$\boldsymbol{X}\boldsymbol{w}^{(1)}$",
            color=cols[0], fontsize=9)
    ax.text(V[1][0] + 0.12, V[1][1] - 0.38, r"$\boldsymbol{X}\boldsymbol{w}^{(2)}$",
            color=cols[1], fontsize=9)
    ax.text(V[2][0] - 0.30, V[2][1] - 0.52, r"$\boldsymbol{X}\boldsymbol{w}^{(3)}$",
            color=cols[2], fontsize=9)
    ax.scatter(*y, s=60, color=RED, zorder=6)
    ax.text(y[0] + 0.16, y[1] - 0.05, r"$\boldsymbol{y}$", color=RED, fontsize=11)
    ax.text(0.5, -1.98, "どれが $\\boldsymbol{y}$ に最も近いか", ha="center", fontsize=9, color=INK_T)
    ax.set_xlim(-2.6, 3.2); ax.set_ylim(-2.35, 2.25)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # the map between the two pictures
    fig.subplots_adjust(wspace=0.08)
    from matplotlib.patches import FancyArrowPatch
    fig.add_artist(FancyArrowPatch((0.368, 0.50), (0.447, 0.50),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=18, lw=1.7, color=INK_T))
    fig.text(0.408, 0.565, r"$\boldsymbol{X}\boldsymbol{w}$", fontsize=11,
             ha="center", va="bottom", color=INK_T)
    save(fig, "ch4_rewrite")


def fig_ls_projection():
    """Least squares as a projection, and the Pythagorean reason it is optimal."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))

    P = np.array([[-2.1, -0.75], [2.1, -1.35], [3.0, 0.75], [-1.2, 1.35]])
    o = np.array([0.0, 0.0])
    yh = np.array([1.25, 0.12])                 # the projection, inside the plane
    y = yh + np.array([0.28, 2.05])             # the target, above the plane
    xw = np.array([-0.55, 0.62])                # another point of the plane

    def panel(ax, title):
        ax.add_patch(plt.Polygon(P, closed=True, facecolor="#DCE6F1",
                                 edgecolor=NAVY, lw=1.0, alpha=0.85, zorder=1))
        ax.text(2.1, -1.18, r"$\mathrm{Col}(\boldsymbol{X})$", color=NAVY, fontsize=8.5)
        ax.scatter(*o, s=14, color="k", zorder=6)
        ax.text(-0.34, -0.30, r"$\boldsymbol{0}$", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-2.6, 4.0); ax.set_ylim(-2.0, 2.9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def right_angle(ax, corner, p_in_plane, p_up, size=0.26):
        d1 = p_in_plane - corner; d1 = d1 / np.linalg.norm(d1) * size
        d2 = p_up - corner; d2 = d2 / np.linalg.norm(d2) * size
        ax.plot([corner[0] + d1[0], corner[0] + d1[0] + d2[0], corner[0] + d2[0]],
                [corner[1] + d1[1], corner[1] + d1[1] + d2[1], corner[1] + d2[1]],
                color="k", lw=0.9, zorder=7)

    # ---- left: the projection itself
    ax = axes[0]
    panel(ax, "誤差が列空間と直交する点が最良")
    ax.annotate("", xy=tuple(y), xytext=tuple(o),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.9))
    ax.annotate("", xy=tuple(yh), xytext=tuple(o),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.7))
    ax.annotate("", xy=tuple(y), xytext=tuple(yh),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.7))
    right_angle(ax, yh, o, y)
    ax.text(y[0] + 0.12, y[1] - 0.05, r"$\boldsymbol{y}$", color=RED, fontsize=10)
    ax.text(yh[0] - 0.30, yh[1] - 0.62,
            r"$\hat{\boldsymbol{y}} = \boldsymbol{X}\boldsymbol{w}^*$", color=NAVY, fontsize=9)
    ax.text(yh[0] + 0.22, (y[1] + yh[1]) / 2 - 0.1,
            r"$\boldsymbol{y}-\hat{\boldsymbol{y}}$", color=GREEN, fontsize=9)

    # ---- right: any other point of the plane gives a longer error
    ax = axes[1]
    panel(ax, "三平方の定理：他の点では必ず遠くなる")
    ax.plot([yh[0], y[0]], [yh[1], y[1]], color=GREEN, lw=1.7, zorder=4)
    ax.plot([xw[0], yh[0]], [xw[1], yh[1]], color=NAVY, lw=1.7, zorder=4)
    ax.plot([xw[0], y[0]], [xw[1], y[1]], color=RED, lw=1.9, zorder=4)
    right_angle(ax, yh, xw, y)
    for p, c in ((y, RED), (yh, GREEN), (xw, NAVY)):
        ax.scatter(*p, s=24, color=c, zorder=6)
    ax.text(y[0] + 0.12, y[1] - 0.05, r"$\boldsymbol{y}$", color=RED, fontsize=10)
    ax.text(yh[0] + 0.10, yh[1] - 0.48, r"$\hat{\boldsymbol{y}}$", color=GREEN, fontsize=10)
    ax.text(xw[0] - 0.92, xw[1] + 0.12, r"$\boldsymbol{X}\boldsymbol{w}$", color=NAVY, fontsize=10)
    ax.text(-2.5, -1.85,
            r"$\|\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}\|^2 = "
            r"\|\boldsymbol{y}-\hat{\boldsymbol{y}}\|^2 + "
            r"\|\hat{\boldsymbol{y}}-\boldsymbol{X}\boldsymbol{w}\|^2$", fontsize=8.5)

    save(fig, "ch4_projection")


def _toy():
    """The running example: D = {(1,3), (2,5), (3,6)}."""
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([3.0, 5.0, 6.0])
    return X, y


def fig_ls_gradient():
    """The loss surface of the running example and the point where the gradient is 0."""
    X, y = _toy()
    w0 = np.linspace(-1.0, 4.4, 320)
    w1 = np.linspace(-0.2, 3.2, 320)
    W0, W1 = np.meshgrid(w0, w1)
    L = np.zeros_like(W0)
    for k in range(3):
        L += (y[k] - X[k, 0] * W0 - X[k, 1] * W1) ** 2
    L /= 3.0

    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    cs = ax.contour(W0, W1, L, levels=[0.1, 0.4, 1.0, 2.2, 4.5, 8.0], colors=NAVY,
                    linewidths=0.8, alpha=0.7)
    ax.clabel(cs, fmt="%g", fontsize=6.5, inline=True)

    opt = np.array([5 / 3, 1.5])
    G = X.T @ X
    for p in (np.array([0.4, 2.6]), np.array([3.4, 0.5]), np.array([0.2, 0.6])):
        g = (2.0 / 3.0) * (G @ p - X.T @ y)
        g = g / np.linalg.norm(g) * 0.75
        ax.annotate("", xy=tuple(p - g), xytext=tuple(p),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
        ax.scatter(*p, s=16, color=RED, zorder=5)
    ax.scatter(*opt, marker="*", s=150, color="k", zorder=6)
    ax.text(opt[0] + 0.18, opt[1] + 0.30,
            r"$\boldsymbol{w}^* = (5/3,\; 3/2)^\top$", fontsize=8.5,
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(0.03, 0.03, r"赤矢印：$-\nabla L_D$（最も速く減る向き）",
            transform=ax.transAxes, fontsize=8, color=RED)
    ax.set_xlabel(r"$w_0$", fontsize=9); ax.set_ylabel(r"$w_1$", fontsize=9)
    ax.set_title(r"$L_D$ の等高線と $\nabla L_D = \boldsymbol{0}$ の点", fontsize=9.5)
    ax.set_xlim(-1.0, 4.4); ax.set_ylim(-0.2, 3.2)
    ax.grid(alpha=0.25)
    save(fig, "ch4_gradient")


def fig_pseudoinverse():
    """Rank deficiency: a whole line of solutions, and the shortest one."""
    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    t = np.linspace(-2.0, 2.0, 2)
    wp = np.array([0.6, 1.2])
    v = np.array([-2.0, 1.0]) / np.sqrt(5)          # direction of Xv = 0
    line = np.array([wp + s * v for s in (-3.2, 3.2)])
    ax.plot(line[:, 0], line[:, 1], color=NAVY, lw=1.8, zorder=3)
    ax.text(line[1][0] + 0.12, line[1][1] - 0.25, "解の集合", color=NAVY, fontsize=8.5)

    for r in (0.6, 1.0, 1.3416, 1.9):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=GRAY, lw=0.7,
                                ls=":", alpha=0.8, zorder=1))
    ax.annotate("", xy=tuple(wp), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0))
    ax.scatter(*wp, s=46, color=RED, zorder=6)
    ax.text(wp[0] + 0.14, wp[1] + 0.06,
            r"$\boldsymbol{w}^+ = (0.6,\; 1.2)^\top$", color=RED, fontsize=8.5)
    for s in (-1.4, 1.4):
        p = wp + s * v
        ax.scatter(*p, s=24, color=NAVY, zorder=5)
        ax.annotate("", xy=tuple(p), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=0.9, ls=":"))
    n = wp / np.linalg.norm(wp)          # the shortest solution is perpendicular
    sz = 0.22                            # to the direction of the solution line
    c1, c2 = wp - v * sz, wp - n * sz
    ax.plot([c1[0], c1[0] - n[0] * sz, c2[0]], [c1[1], c1[1] - n[1] * sz, c2[1]],
            color="k", lw=0.9, zorder=7)
    ax.scatter(0, 0, s=22, color="k", zorder=6)
    ax.text(-0.33, -0.3, r"$\boldsymbol{0}$", fontsize=9)
    ax.text(0.03, 0.03, "どの点も予測は同じ\n原点に最も近いのが擬似逆行列の解",
            transform=ax.transAxes, fontsize=8, color="#333333")
    ax.set_xlabel(r"$w_1$", fontsize=9); ax.set_ylabel(r"$w_2$", fontsize=9)
    ax.set_title("ランク落ち：解は直線状に無数にある", fontsize=9.5)
    ax.set_xlim(-2.6, 3.4); ax.set_ylim(-1.2, 3.2)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    save(fig, "ch4_pseudoinverse")


def fig_gradient_concept():
    """Partial derivatives as slopes of slices, and the gradient as a vector normal to contours."""
    def f(a, b):
        return a ** 2 + 2 * b ** 2

    p = np.array([1.0, 0.5])                        # the point we look at; grad f = (2, 2)
    fig = plt.figure(figsize=(8.0, 3.6))

    # --- left: the surface, sliced along each axis through p
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    g = np.linspace(-1.6, 1.6, 60)
    A, B = np.meshgrid(g, g)
    ax.plot_surface(A, B, f(A, B), color="#C9D7EA", alpha=0.3, lw=0, antialiased=True)
    ax.plot_wireframe(A, B, f(A, B), rstride=6, cstride=6, color=NAVY, lw=0.25, alpha=0.25)
    # slices through p; lifted slightly so the surface does not hide them
    t = np.linspace(-1.6, 1.6, 80)
    ax.plot(t, np.full_like(t, p[1]), f(t, p[1]) + 0.04, color=NAVY, lw=2.2)     # w2 fixed
    ax.plot(np.full_like(t, p[0]), t, f(p[0], t) + 0.04, color=RED, lw=2.2)      # w1 fixed
    s = np.linspace(-0.8, 0.8, 2)
    z0 = f(*p) + 0.04
    ax.plot(p[0] + s, np.full_like(s, p[1]), z0 + 2 * s, color=NAVY, lw=1.6, ls="--")
    ax.plot(np.full_like(s, p[0]), p[1] + s, z0 + 2 * s, color=RED, lw=1.6, ls="--")
    ax.scatter(*p, z0, s=34, color="k", depthshade=False)
    ax.text2D(0.02, 0.93, r"青：$w_2 = 0.5$ で切った断面，傾き $\partial f/\partial w_1 = 2w_1 = 2$",
              transform=ax.transAxes, color=NAVY, fontsize=7.5)
    ax.text2D(0.02, 0.87, r"赤：$w_1 = 1$ で切った断面，傾き $\partial f/\partial w_2 = 4w_2 = 2$",
              transform=ax.transAxes, color=RED, fontsize=7.5)
    ax.set_xlabel(r"$w_1$", fontsize=8, labelpad=-4)
    ax.set_ylabel(r"$w_2$", fontsize=8, labelpad=-4)
    ax.set_zlabel(r"$f$", fontsize=8, labelpad=-6)
    ax.tick_params(labelsize=6, pad=-2)
    ax.set_zlim(0, 8)
    ax.view_init(elev=24, azim=-48)
    ax.set_title(r"偏微分 ＝ 軸に沿った断面の傾き", fontsize=9.5)

    # --- right: contours and the gradient vectors
    ax = fig.add_subplot(1, 2, 2)
    g = np.linspace(-1.8, 1.8, 300)
    A, B = np.meshgrid(g, g)
    cs = ax.contour(A, B, f(A, B), levels=[0.25, 0.75, 1.5, 2.5, 4.0, 6.0],
                    colors=NAVY, linewidths=0.8, alpha=0.7)
    ax.clabel(cs, fmt="%g", fontsize=6.5, inline=True)
    k = 0.2                                         # display scale of the arrows
    for q in (np.array([-1.2, 0.3]), np.array([0.5, -0.8]), np.array([-0.6, -0.5]),
              np.array([1.3, -0.3]), np.array([0.2, 0.9]), np.array([-0.3, 0.2])):
        grad = np.array([2 * q[0], 4 * q[1]])
        ax.annotate("", xy=tuple(q + k * grad), xytext=tuple(q),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.3))
        ax.scatter(*q, s=12, color=GRAY, zorder=5)
    grad = np.array([2.0, 2.0])
    ax.annotate("", xy=tuple(p + k * grad), xytext=tuple(p),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2.0), zorder=7)
    ax.annotate("", xy=(p[0] + k * grad[0], p[1]), xytext=tuple(p),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.3, ls="--"), zorder=6)
    ax.annotate("", xy=(p[0] + k * grad[0], p[1] + k * grad[1]),
                xytext=(p[0] + k * grad[0], p[1]),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3, ls="--"), zorder=6)
    ax.scatter(*p, s=30, color="k", zorder=8)
    ax.text(p[0] + 0.05, p[1] - 0.32, r"$\boldsymbol{w} = (1,\; 0.5)^\top$", fontsize=8,
            ha="right", bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(p[0] + 0.3, p[1] + 0.55, r"$\nabla f(\boldsymbol{w}) = (2,\; 2)^\top$",
            fontsize=8.5, bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(p[0] + 0.14, p[1] - 0.05, r"$2$", color=NAVY, fontsize=8, va="top")
    ax.text(p[0] + k * grad[0] + 0.05, p[1] + 0.15, r"$2$", color=RED, fontsize=8)
    ax.text(0.03, 0.03, "矢印は等高線に直交し、等高線が密なほど長い",
            transform=ax.transAxes, fontsize=7.5, color="#333333")
    ax.set_xlabel(r"$w_1$", fontsize=9); ax.set_ylabel(r"$w_2$", fontsize=9)
    ax.set_title(r"勾配 $\nabla f$ ＝ 最も急に増える向き", fontsize=9.5)
    ax.set_xlim(-1.8, 2.2); ax.set_ylim(-1.9, 1.9)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    fig.suptitle(r"$f(w_1, w_2) = w_1^2 + 2w_2^2$", fontsize=10, y=1.0)
    save(fig, "ch4_gradient_concept")


def fig_definiteness():
    """Positive definite vs semi-definite: bowl/trough, the angle between v and Av, eigen-directions."""
    mats = [(np.diag([1.0, 2.0]), r"\boldsymbol{A}", "正定値"),
            (np.diag([1.0, 0.0]), r"\boldsymbol{B}", "半正定値")]
    fig = plt.figure(figsize=(9.6, 6.2))
    g = np.linspace(-1.5, 1.5, 70)
    V1, V2 = np.meshgrid(g, g)
    for r, (M, name, kind) in enumerate(mats):
        Q = M[0, 0] * V1 ** 2 + M[1, 1] * V2 ** 2
        form = (r"$v_1^2 + 2v_2^2$" if r == 0 else r"$v_1^2$")

        # --- column 1: the surface of the quadratic form
        ax = fig.add_subplot(2, 3, 3 * r + 1, projection="3d")
        ax.plot_surface(V1, V2, Q, color="#C9D7EA", alpha=0.45, lw=0, antialiased=True)
        ax.plot_wireframe(V1, V2, Q, rstride=7, cstride=7, color=NAVY, lw=0.3, alpha=0.5)
        if r == 1:                                   # the flat direction of the trough
            t = np.linspace(-1.5, 1.5, 2)
            ax.plot(np.zeros(2), t, np.zeros(2) + 0.02, color=RED, lw=2.2)
            ax.text2D(0.02, 0.88, r"赤：$v_2$ 方向に動いても値が変わらない",
                      transform=ax.transAxes, color=RED, fontsize=7.5)
        ax.set_xlabel(r"$v_1$", fontsize=8, labelpad=-5)
        ax.set_ylabel(r"$v_2$", fontsize=8, labelpad=-5)
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_zlim(0, 4.5)
        ax.view_init(elev=24, azim=-55)
        ax.set_title(("お椀：" if r == 0 else "樋：") + r"$\boldsymbol{v}^\top " + name + r"\boldsymbol{v} = $" + form,
                     fontsize=9)

        # --- column 2: v and Mv never make an angle of 90 degrees or more
        ax = fig.add_subplot(2, 3, 3 * r + 2)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GRAY, lw=0.8, ls=":"))
        for deg in (20, 55, 90):
            v = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
            Mv = M @ v
            ax.annotate("", xy=tuple(v), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6))
            if np.linalg.norm(Mv) > 1e-9:
                ax.annotate("", xy=tuple(Mv), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6, ls="--"))
            else:
                ax.scatter(0, 0, s=60, color=RED, zorder=6)
                ax.text(-2.2, -0.45, r"$\boldsymbol{v} = (0,1)^\top \mapsto " + name + r"\boldsymbol{v} = \boldsymbol{0}$",
                        color=RED, fontsize=8)
        ax.text(0.78, 0.5, r"$\boldsymbol{v}$", color=NAVY, fontsize=9)
        if r == 0:
            ax.text(0.15, 2.05, r"$\boldsymbol{A}\boldsymbol{v}$", color=RED, fontsize=9)
            ax.text(-2.3, -2.1, r"どの $\boldsymbol{v}$ も $\boldsymbol{A}\boldsymbol{v}$ となす角は $90^\circ$ 未満",
                    fontsize=7.5, color="#333333")
        else:
            ax.text(1.0, 0.1, r"$\boldsymbol{B}\boldsymbol{v}$", color=RED, fontsize=9)
            ax.text(-2.3, -2.1, r"$\boldsymbol{v} = (0,1)^\top$ はつぶされて $\boldsymbol{0}$ になる",
                    fontsize=7.5, color="#333333")
        ax.axhline(0, color=GRAY, lw=0.6); ax.axvline(0, color=GRAY, lw=0.6)
        ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal")
        ax.set_xlabel(r"$v_1$", fontsize=8); ax.set_ylabel(r"$v_2$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(r"$\boldsymbol{v}$（青）と $" + name + r"\boldsymbol{v}$（赤）のなす角", fontsize=9)

        # --- column 3: contours of the quadratic form and the eigen-directions
        ax = fig.add_subplot(2, 3, 3 * r + 3)
        cs = ax.contour(V1, V2, Q, levels=[0.25, 0.75, 1.5, 2.5], colors=NAVY,
                        linewidths=0.8, alpha=0.7)
        ax.clabel(cs, fmt="%g", fontsize=6.5, inline=True)
        for lam, e, col, off in ((M[0, 0], np.array([1.0, 0.0]), NAVY, (0.05, -0.3)),
                                 (M[1, 1], np.array([0.0, 1.0]), RED, (0.08, 0.0))):
            ax.annotate("", xy=tuple(e * 1.2), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
            ax.text(e[0] * 1.2 + off[0], e[1] * 1.2 + off[1],
                    r"$\lambda = %g$" % lam, color=col, fontsize=8.5)
        if r == 1:
            ax.text(-1.45, -1.35, r"固有値 $0$ の向き（赤）に沿って値が一定",
                    fontsize=7.5, color="#333333")
        else:
            ax.text(-1.45, -1.35, r"固有値がすべて正：等高線は閉じた楕円",
                    fontsize=7.5, color="#333333")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
        ax.set_xlabel(r"$v_1$", fontsize=8); ax.set_ylabel(r"$v_2$", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
        ax.set_title(r"等高線と固有ベクトル", fontsize=9)

        fig.text(0.005, 0.74 - 0.48 * r,
                 r"$" + name + r" = \mathrm{diag}(1,\, %d)$" % (2 - 2 * r)
                 + "\n" + kind, fontsize=9.5, ha="left", va="center",
                 color=(GREEN if r == 0 else RED), fontweight="bold")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.06, wspace=0.3, hspace=0.35)
    fig.savefig(OUT / "ch4_definiteness.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ch4_definiteness.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("  figures/ch4_definiteness.pdf + .png")


if __name__ == "__main__":
    print("generating figures...")
    fig_fit(); fig_error_curve()
    fig_vec_ops(); fig_projection(); fig_cossim(); fig_attention(); fig_neuron()
    fig_neuron_diagram(); fig_layer_diagram(); fig_attention_flow(); fig_embedding()
    fig_span(); fig_independence(); fig_linearmap(); fig_matmul(); fig_colspace_sweep(); fig_colspace(); fig_flat_valley()
    fig_eigen(); fig_eigen_ml()
    fig_param_to_distance(); fig_ls_projection()
    fig_gradient_concept(); fig_definiteness(); fig_ls_gradient(); fig_pseudoinverse()
