# Copyright (c) 2026, fast-diffusion-attention authors.
"""Figures of the README and of docs/: the attention pattern, the tiling and the performance charts.

The performance charts are drawn from the tables of docs/performance.md, the single source of the numbers. Every
figure is written for GitHub's light and dark themes: assets/<name>.png and assets/<name>_dark.png.

Usage: python benchmarks/plot_figures.py [--only pattern tiles ...] [--font-dir DIR]
Requires matplotlib and numpy; uses the Inter font if it is installed (or found in --font-dir).
"""

import argparse
import io
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PERF_DOC = ROOT / "docs" / "performance.md"
OUT_DIR = ROOT / "assets"
DPI = 200
WIDTH = 12  # inches; README images are displayed about 880 px wide, so 1 pt of text is about 1 px

THEMES = {
    "light": dict(fg="#1f2328", muted="#59636e", faint="#8c959f", grid="#e1e6eb", card="#ffffff",
                  card_edge="#d1d9e0", baseline="#b4bcc5", masked="#f0f3f6", masked_edge="#e1e6eb",
                  noisy="#f97316", cross="#8b5cf6", clean="#0ea5e9",
                  bs={4: "#a78bfa", 16: "#7c3aed", 64: "#4c1d95", 256: "#7c3aed", 1024: "#4c1d95"},
                  gpus=("#0d9488", "#4f46e5", "#d97706")),
    "dark": dict(fg="#e6edf3", muted="#9198a1", faint="#6e7681", grid="#262c33", card="#0d1117",
                 card_edge="#30363d", baseline="#4b535d", masked="#161b22", masked_edge="#21262d",
                 noisy="#fb923c", cross="#a78bfa", clean="#38bdf8",
                 bs={4: "#ddd6fe", 16: "#a78bfa", 64: "#7c3aed", 256: "#a78bfa", 1024: "#7c3aed"},
                 gpus=("#2dd4bf", "#818cf8", "#fbbf24")),
}


# ----------------------------------------------------------------------------------------------------------------------
# Tables of docs/performance.md


def read_tables(path=PERF_DOC):
    """{heading: rows} with the table rows under every heading (lists of cells, header row first)."""
    tables, heading = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        elif line.startswith("|") and heading is not None:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                tables.setdefault(heading, []).append(cells)
    return tables


def table(tables, prefix):
    for heading, rows in tables.items():
        if heading.startswith(prefix):
            return rows
    raise KeyError(f"no table under a heading starting with {prefix!r} in {PERF_DOC}")


def ms_tflops(cell):
    m = re.fullmatch(r"([\d.]+) \((\d+)\)", cell)
    return float(m.group(1)), float(m.group(2))


def timing_rows(rows):
    """Rows of a 'block size | forward ms (TFLOPS) | baseline | rel. | [backward ...]' table, labels filled down."""
    out, label = [], None
    for cells in rows[1:]:
        label = cells[0] or label
        row = dict(label=label, bs=int(cells[1]), fwd=ms_tflops(cells[2]), base_fwd=ms_tflops(cells[3]))
        if len(cells) >= 8:
            row.update(bwd=ms_tflops(cells[5]), base_bwd=ms_tflops(cells[6]))
        out.append(row)
    return out


def step_tflops(fwd, bwd):
    """TFLOPS of forward + backward from the (ms, TFLOPS) of each."""
    return (fwd[0] * fwd[1] + bwd[0] * bwd[1]) / (fwd[0] + bwd[0])


def length_label(label):
    return re.search(r"L = (\w+)", label).group(1)


# ----------------------------------------------------------------------------------------------------------------------
# Drawing helpers (layout in inches from the top-left corner of the figure)


def setup_fonts(font_dir=None):
    if font_dir:
        for f in Path(font_dir).glob("**/*.ttf"):
            font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Inter", "Helvetica Neue", "Arial") if f in names), "DejaVu Sans")
    plt.rcParams.update({"font.family": family, "font.size": 11, "mathtext.fontset": "custom",
                         "mathtext.it": f"{family}:italic", "mathtext.rm": family,
                         "mathtext.cal": family})


def new_figure(th, height, **subplots):
    """Figure on a rounded card (the corners stay transparent)."""
    if subplots:
        fig, axes = plt.subplots(figsize=(WIDTH, height), **subplots)
    else:
        fig, axes = plt.figure(figsize=(WIDTH, height)), None
    fig.patches.append(FancyBboxPatch((0.02, 0.02), WIDTH - 0.04, height - 0.04,
                                      boxstyle="round,pad=0,rounding_size=0.16", transform=fig.dpi_scale_trans,
                                      facecolor=th["card"], edgecolor=th["card_edge"], linewidth=1.0, zorder=-10))
    return fig, axes


def canvas(fig):
    """Axes covering the figure, in inches, y pointing down."""
    w, h = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return ax


def header(fig, th, title, subtitle=None):
    h = fig.get_size_inches()[1]
    fig.text(0.34 / WIDTH, 1 - 0.32 / h, title, fontsize=15, fontweight="bold", color=th["fg"], va="top")
    if subtitle:
        fig.text(0.34 / WIDTH, 1 - 0.68 / h, subtitle, fontsize=10.5, color=th["muted"], va="top")


def place_axes(fig, top, bottom=0.62, left=0.72, right=0.24, wspace=0.2):
    """Subplot area with margins in inches."""
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=left / w, right=1 - right / w, top=1 - top / h, bottom=bottom / h, wspace=wspace)


def legend(fig, th, ax, top=1.08, extra=()):
    h = fig.get_size_inches()[1]
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles + [e[0] for e in extra], labels + [e[1] for e in extra]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.3 / WIDTH, 1 - top / h), ncol=len(labels),
               frameon=False, fontsize=10.5, handlelength=1.0, handleheight=1.0, columnspacing=1.8,
               labelcolor=th["fg"], borderaxespad=0)


def style_axes(ax, th, ylabel=None):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["faint"])
    ax.tick_params(axis="both", colors=th["muted"], labelsize=10, length=0, pad=6)
    ax.grid(axis="y", color=th["grid"], linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor("none")
    if ylabel:
        ax.set_ylabel(ylabel, color=th["muted"], fontsize=10)


def panel_title(ax, th, text, sub=None):
    ax.text(0, 1.0, text, transform=ax.transAxes, fontsize=12, fontweight="bold", color=th["fg"], va="bottom")
    if sub:
        ax.text(0, 1.0, sub, transform=ax.transAxes, fontsize=9.5, color=th["muted"], va="top")


def grouped_bars(ax, th, groups, series, label_size=8.5):
    """series: [(name, color, values)], one value per group; values written vertically above the bars."""
    k = len(series)
    width = 0.84 / k
    x = np.arange(len(groups))
    top = max(v for _, _, vals in series for v in vals)
    for i, (name, color, vals) in enumerate(series):
        xs = x - 0.42 + width * (i + 0.5)
        ax.bar(xs, vals, width * 0.86, color=color, label=name, zorder=3, linewidth=0)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + top * 0.02, f"{v:.0f}", ha="center", va="bottom", fontsize=label_size, color=th["fg"],
                    rotation=90, zorder=4)
    ax.set_xticks(x, groups)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(0, top * 1.24)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))


def save(fig, name, theme):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}{'' if theme == 'light' else '_dark'}.png"
    buf = io.BytesIO()
    fig.savefig(buf, dpi=DPI, transparent=True)
    plt.close(fig)
    # Flat colors and text: a 256-color palette looks the same and makes the files 3-4x smaller
    image = Image.open(buf).quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    image.save(path, optimize=True)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# Diagrams


def mask_regions(n, block_size):
    """(noisy->noisy, noisy->clean, clean->clean) boolean (2n, 2n) masks, as fast_diffusion_attention.reference."""
    idx = np.arange(2 * n)
    clean = idx >= n
    blk = np.where(clean, idx - n, idx) // block_size
    qc, kc, qb, kb = clean[:, None], clean[None, :], blk[:, None], blk[None, :]
    return ~qc & ~kc & (qb == kb), ~qc & kc & (kb < qb), qc & kc & (kb <= qb)


def cell(ax, x, y, size, face, edge=None, lw=0.0, r=0.07):
    ax.add_patch(FancyBboxPatch((x, y), size, size, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=face,
                                edgecolor=edge or face, linewidth=lw, zorder=2))


def masked_cell(ax, th, x, y, size, r=0.07):
    cell(ax, x, y, size, th["masked"], th["masked_edge"], 0.8, r)


def block_matrix(ax, th, nb, x0, y0, size, gap, half_gap):
    """Block-level diffusion mask with nb blocks per half, top-left corner at (x0, y0). Returns the cell position."""
    nn, nc, cc = mask_regions(nb, 1)

    def pos(i):
        return i * (size + gap) + (half_gap if i >= nb else 0)

    for r in range(2 * nb):
        for c in range(2 * nb):
            color = th["noisy"] if nn[r, c] else th["cross"] if nc[r, c] else th["clean"] if cc[r, c] else None
            if color:
                cell(ax, x0 + pos(c), y0 + pos(r), size, color)
            else:
                masked_cell(ax, th, x0 + pos(c), y0 + pos(r), size)
    return pos


def legend_item(ax, th, x, y, swatch, head, sub):
    swatch(x, y - 0.02)
    ax.text(x + 0.55, y + 0.02, head, fontsize=12, color=th["fg"], va="top", fontweight="semibold")
    ax.text(x + 0.55, y + 0.32, sub, fontsize=10.5, color=th["muted"], va="top")


def fig_pattern(th):
    """The attention pattern: block-level mask and its rules (README hero)."""
    fig, _ = new_figure(th, 6.3)
    ax = canvas(fig)
    nb, size, gap, half_gap = 4, 0.5, 0.06, 0.2
    x0, y0 = 1.55, 1.45
    pos = block_matrix(ax, th, nb, x0, y0, size, gap, half_gap)
    span = nb * (size + gap) - gap
    for i in range(2 * nb):
        name = f"{i % nb + 1}"
        ax.text(x0 + pos(i) + size / 2, y0 - 0.1, name, ha="center", va="bottom", fontsize=10, color=th["faint"])
        ax.text(x0 - 0.12, y0 + pos(i) + size / 2, name, ha="right", va="center", fontsize=10, color=th["faint"])
    for h, (text, color) in enumerate([("noisy  $x_t$", th["noisy"]), ("clean  $x_0$", th["clean"])]):
        s = pos(h * nb)
        ax.plot([x0 + s, x0 + s + span], [y0 - 0.48, y0 - 0.48], color=th["faint"], lw=1.2, solid_capstyle="round")
        ax.text(x0 + s + span / 2, y0 - 0.58, text, ha="center", va="bottom", fontsize=12.5, color=th["fg"],
                fontweight="semibold")
        ax.plot([x0 - 0.46, x0 - 0.46], [y0 + s, y0 + s + span], color=th["faint"], lw=1.2, solid_capstyle="round")
        ax.text(x0 - 0.56, y0 + s + span / 2, text, ha="right", va="center", fontsize=12.5, color=th["fg"],
                fontweight="semibold", rotation=90)
    ax.text(0.34, 0.3, "rows: query blocks\ncolumns: key blocks", fontsize=9.5, color=th["faint"], va="top",
            linespacing=1.4)

    tx = 6.85
    ax.text(tx, 0.55, "Block diffusion attention", fontsize=19, fontweight="bold", color=th["fg"], va="top")
    ax.text(tx, 1.08, "A sample of 2n tokens is [ noisy $x_t$ | clean $x_0$ ], each half\n"
                      "split into blocks of block_size tokens.", fontsize=11, color=th["muted"], va="top",
            linespacing=1.55)
    rules = [(th["noisy"], "A noisy block sees itself", "fully bidirectional within the block"),
             (th["cross"], "... and all the previous clean blocks", "the clean context it is denoised from"),
             (th["clean"], "A clean block sees itself and the previous", "clean blocks: block-causal attention"),
             (None, "Everything else is masked", "and never computed")]
    y = 2.2
    for color, head, sub in rules:
        if color:
            swatch = lambda x, yy, c=color: cell(ax, x, yy, 0.32, c, r=0.06)  # noqa: E731
        else:
            swatch = lambda x, yy: masked_cell(ax, th, x, yy, 0.32, r=0.06)  # noqa: E731
        legend_item(ax, th, tx, y, swatch, head, sub)
        y += 0.8
    ax.text(tx, y + 0.1, "About 25% of the 2n × 2n matrix is visible: the work of\n"
                         "causal attention on each half.", fontsize=10.5, color=th["fg"], va="top", linespacing=1.55,
            style="italic")
    return fig


def tile_classes(n, block_size, tile):
    """Per tile of the (2n, 2n) mask: 0 invisible, 1 partially visible, 2 fully visible."""
    vis = np.logical_or.reduce(mask_regions(n, block_size))
    t = 2 * n // tile
    v = vis.reshape(t, tile, t, tile).transpose(0, 2, 1, 3).reshape(t, t, -1)
    return np.where(v.all(-1), 2, np.where(v.any(-1), 1, 0))


def fig_tiles(th):
    """What the kernels compute: the tiles that intersect the mask, masking only the partially visible ones."""
    n, bs, tile = 512, 32, 128
    classes = tile_classes(n, bs, tile)
    t = classes.shape[0]
    fig, _ = new_figure(th, 6.2)
    ax = canvas(fig)
    x0, y0, size = 1.1, 1.0, 4.75
    ts, pad = size / t, 0.035
    rgba = np.zeros((2 * n, 2 * n, 4))
    for region, key in zip(mask_regions(n, bs), ("noisy", "cross", "clean")):
        rgba[region] = matplotlib.colors.to_rgba(th[key])
    dash = (0, (2.2, 1.6))
    for r in range(t):
        for c in range(t):
            x, y = x0 + c * ts + pad, y0 + r * ts + pad
            s = ts - 2 * pad
            if classes[r, c] == 0:
                ax.add_patch(Rectangle((x, y), s, s, facecolor=th["masked"], edgecolor="none", zorder=1))
                continue
            ax.imshow(rgba[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile], extent=(x, x + s, y + s, y),
                      interpolation="nearest", zorder=2)
            if classes[r, c] == 1:
                ax.add_patch(Rectangle((x, y), s, s, facecolor="none", edgecolor=th["fg"], linewidth=1.3,
                                       linestyle=dash, zorder=3))
    for h, name in enumerate(("noisy", "clean")):
        ax.text(x0 + (2 * h + 1) * size / 4, y0 - 0.16, f"{name} keys", ha="center", va="bottom", fontsize=11,
                color=th["muted"])
        ax.text(x0 - 0.16, y0 + (2 * h + 1) * size / 4, f"{name} queries", ha="right", va="center", fontsize=11,
                color=th["muted"], rotation=90)

    tx = 6.6
    computed, masked = int((classes > 0).sum()), int((classes == 1).sum())
    ax.text(tx, 0.55, "Only the visible tiles are computed", fontsize=17, fontweight="bold", color=th["fg"],
            va="top")
    ax.text(tx, 1.02, f"One sample of 2n = {2 * n} tokens, block size {bs}, {tile} × {tile} tiles.", fontsize=11,
            color=th["muted"], va="top")

    def swatch_full(x, y):
        cell(ax, x, y, 0.32, th["cross"], r=0.03)

    def swatch_masked(x, y):
        cell(ax, x, y, 0.32, th["masked"], r=0.03)
        ax.add_patch(Rectangle((x, y + 0.16), 0.16, 0.16, facecolor=th["cross"], edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((x, y), 0.32, 0.32, facecolor="none", edgecolor=th["fg"], linewidth=1.3,
                               linestyle=dash, zorder=3))

    def swatch_skipped(x, y):
        cell(ax, x, y, 0.32, th["masked"], r=0.03)

    y = 1.7
    for swatch, head, sub in [(swatch_full, "Fully visible tile", "computed without any masking"),
                              (swatch_masked, "Partially visible tile", "crosses a block boundary: masked"),
                              (swatch_skipped, "Invisible tile", "skipped: never loaded, never computed")]:
        legend_item(ax, th, tx, y, swatch, head, sub)
        y += 0.78
    dense, causal = t * t, t * (t + 1) // 2
    y += 0.1
    ax.text(tx, y, "Tiles computed for this sample", fontsize=12, color=th["fg"], fontweight="semibold", va="top")
    y += 0.4
    scale = 2.5 / dense
    for name, count, color in [("dense mask", dense, th["baseline"]),
                               ("causal attention on 2n tokens", causal, th["baseline"]),
                               ("diffusion attention", computed, th["cross"])]:
        ax.add_patch(FancyBboxPatch((tx + 2.3, y), count * scale, 0.26, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=color, edgecolor="none"))
        ax.text(tx + 2.2, y + 0.13, name, ha="right", va="center", fontsize=10, color=th["muted"])
        ax.text(tx + 2.3 + count * scale + 0.08, y + 0.13, f"{count}", ha="left", va="center", fontsize=10,
                color=th["fg"], fontweight="semibold")
        y += 0.36
    ax.text(tx, y + 0.06, f"{masked} of the {computed} tiles are masked here; with 2n = 16K tokens, under 5%.",
            fontsize=10, color=th["muted"], va="top", style="italic")
    return fig


def fig_equal_work(th):
    """The rows of each half form a block-causal staircase: the work of causal attention on the two halves."""
    fig, _ = new_figure(th, 5.55)
    ax = canvas(fig)
    nb, size, gap, half_gap = 4, 0.38, 0.05, 0.3
    x0, y0 = 1.35, 1.55
    pos = block_matrix(ax, th, nb, x0, y0, size, gap, half_gap)
    span = nb * (size + gap) - gap
    right = x0 + pos(2 * nb - 1) + size
    ax.text((x0 + right) / 2, y0 - 0.14, "diffusion mask", ha="center", va="bottom", fontsize=11.5,
            color=th["fg"], fontweight="semibold")
    sx = right + 1.3
    for h, title in enumerate(("noisy query rows", "clean query rows")):
        top = y0 + pos(h * nb)
        for r in range(nb):
            for c in range(r + 1):
                color = th["clean"] if h else (th["noisy"] if c == r else th["cross"])
                cell(ax, sx + c * (size + gap), top + r * (size + gap), size, color)
        ax.add_patch(FancyArrowPatch((right + 0.25, top + span / 2), (sx - 0.3, top + span / 2),
                                     arrowstyle="-|>,head_length=6,head_width=3.5", color=th["faint"], lw=1.4,
                                     mutation_scale=1.6))
        ax.text(sx + span + 0.4, top + span / 2 - 0.08, title, fontsize=12, color=th["fg"], fontweight="semibold",
                va="bottom")
        ax.text(sx + span + 0.4, top + span / 2 + 0.06, "= causal attention on n tokens", fontsize=11,
                color=th["muted"], va="top")
    header(fig, th, "The same work as causal attention on the two halves",
           "Rearranged, each half is a block-causal staircase: causal FlashAttention on the 2B halves of length n "
           "does the work of B samples of length 2n.")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Performance charts


def fig_h800_bf16(th, tables):
    rows = timing_rows(table(tables, "BF16, head dimension 128"))
    lengths = list(dict.fromkeys(length_label(r["label"]) for r in rows))
    by = {(length_label(r["label"]), r["bs"]): r for r in rows}
    fig, axes = new_figure(th, 4.75, nrows=1, ncols=3)
    place_axes(fig, top=1.72, bottom=0.62)
    parts = [("Forward", lambda r: r["fwd"][1], lambda r: r["base_fwd"][1]),
             ("Backward", lambda r: r["bwd"][1], lambda r: r["base_bwd"][1]),
             ("Forward + backward", lambda r: step_tflops(r["fwd"], r["bwd"]),
              lambda r: step_tflops(r["base_fwd"], r["base_bwd"]))]
    for ax, (name, ours, base) in zip(axes, parts):
        series = [("FlashAttention-3 (same work)", th["baseline"], [base(by[(L, 4)]) for L in lengths])]
        series += [(f"FDA, block size {bs}", th["bs"][bs], [ours(by[(L, bs)]) for L in lengths]) for bs in (4, 16, 64)]
        grouped_bars(ax, th, lengths, series)
        style_axes(ax, th, "TFLOPS" if ax is axes[0] else None)
        panel_title(ax, th, name)
    axes[1].set_xlabel("sequence length L = 2n (batch size 16K / L)", color=th["muted"], fontsize=10)
    header(fig, th, "H800: BF16, head dimension 128",
           "TFLOPS on the visible (query, key) pairs, 32 query / 8 KV heads. Baseline: FlashAttention-3 causal "
           "attention on the two halves of the samples.")
    legend(fig, th, axes[0])
    return fig


def fig_h800_more(th, tables):
    fp8 = timing_rows(table(tables, "FP8 forward"))
    varlen = timing_rows(table(tables, "Varlen"))
    hdims = timing_rows(table(tables, "Head dimensions 64 and 256"))
    fig, axes = new_figure(th, 4.8, nrows=1, ncols=4, gridspec_kw=dict(width_ratios=[4, 4, 4, 2.3]))
    place_axes(fig, top=1.95, bottom=0.5, wspace=0.24)

    def panel(ax, rows, key, groups, name, sub, step):
        by = {(key(r), r["bs"]): r for r in rows}
        if step:
            ours = lambda r: step_tflops(r["fwd"], r["bwd"])  # noqa: E731
            base = lambda r: step_tflops(r["base_fwd"], r["base_bwd"])  # noqa: E731
        else:
            ours, base = (lambda r: r["fwd"][1]), (lambda r: r["base_fwd"][1])
        series = [("FlashAttention-3 (same work)", th["baseline"], [base(by[(g, 4)]) for g in groups])]
        series += [(f"FDA, block size {bs}", th["bs"][bs], [ours(by[(g, bs)]) for g in groups]) for bs in (4, 64)]
        grouped_bars(ax, th, groups, series)
        style_axes(ax, th, "TFLOPS" if ax is axes[0] else None)
        panel_title(ax, th, name, sub)

    def short(v):
        return f"{int(v) // 1024}K" if int(v) >= 1024 else v

    def vkey(r):
        return "-".join(short(v) for v in re.findall(r"\d+", r["label"]))

    lkey = lambda r: length_label(r["label"])  # noqa: E731
    panel(axes[0], fp8, lkey, ["2K", "4K", "8K", "16K"], "FP8 forward", "head dim 128, by L", step=False)
    panel(axes[1], varlen, vkey, list(dict.fromkeys(vkey(r) for r in varlen)), "Varlen, fwd + bwd",
          "random sample lengths in a range", step=True)
    panel(axes[2], [r for r in hdims if r["label"].startswith("d = 64")], lkey, ["2K", "4K", "8K", "16K"],
          "Head dim 64, fwd + bwd", "by L", step=True)
    panel(axes[3], [r for r in hdims if r["label"].startswith("d = 256")], lkey, ["4K", "16K"], "Head dim 256",
          "fwd + bwd, by L", step=True)
    header(fig, th, "H800: FP8, varlen batches and other head dimensions",
           "TFLOPS on the visible pairs, batches of 16K tokens, 32 query / 8 KV heads, BF16 unless noted.")
    legend(fig, th, axes[0])
    return fig


def fig_ampere_ada(th, tables):
    """Training step relative to FlashAttention-2 per head dimension on the GPUs of the fa2 branch."""
    rows = table(tables, "Ampere and Ada")
    gpus = rows[0][1:]
    hdims, data = [], []
    for cells in rows[1:]:
        hdims.append(cells[0].split()[0])
        data.append([tuple(float(g) for g in re.fullmatch(r"([\d.]+) - ([\d.]+) \(\*\*([\d.]+)\*\*\)", c).groups())
                     for c in cells[1:]])
    fig, ax = new_figure(th, 4.9, nrows=1, ncols=1)
    place_axes(fig, top=1.55, bottom=0.55, left=0.78)
    x = np.arange(len(hdims))
    step = 0.26
    for i, gpu in enumerate(gpus):
        xs = x + (i - (len(gpus) - 1) / 2) * step
        lo, hi, gm = (np.array([d[i][j] for d in data]) for j in range(3))
        color = th["gpus"][i]
        ax.vlines(xs, lo, hi, color=color, linewidth=7, alpha=0.3, zorder=3, capstyle="round")
        ax.scatter(xs, gm, s=70, color=color, edgecolor=th["card"], linewidth=1.6, zorder=5, label=gpu)
        for xi, v, top in zip(xs, gm, hi):
            ax.text(xi, top + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, color=th["fg"],
                    fontweight="semibold", zorder=6)
    ax.axhline(1, color=th["faint"], linewidth=1.1, zorder=2)
    ax.set_xticks(x, [f"head dim {h}" for h in hdims])
    ax.set_xlim(-0.5, len(hdims) - 0.5)
    lo_all = min(d[i][0] for d in data for i in range(len(gpus)))
    hi_all = max(d[i][1] for d in data for i in range(len(gpus)))
    ax.set_ylim(math.floor(lo_all * 10) / 10, hi_all + 0.12)
    style_axes(ax, th)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    header(fig, th, "Ampere / Ada (fa2 branch): training step vs FlashAttention-2",
           "Forward + backward, BF16, FlashAttention-2's benchmark shapes, L = 512 to 16K, block sizes 4 / 16 / 64. "
           "Dot: geometric mean; bar: range.")
    legend(fig, th, ax, extra=[(Line2D([], [], color=th["faint"], linewidth=1.1), "FlashAttention-2 (1.0x)")])
    return fig


FIGURES = {
    "pattern": lambda th, t: fig_pattern(th),
    "tiles": lambda th, t: fig_tiles(th),
    "equal_work": lambda th, t: fig_equal_work(th),
    "perf_h800_bf16": fig_h800_bf16,
    "perf_h800_more": fig_h800_more,
    "perf_ampere_ada": fig_ampere_ada,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), help="figures to draw (default: all)")
    parser.add_argument("--font-dir", help="directory with extra .ttf fonts (e.g. Inter)")
    args = parser.parse_args()
    setup_fonts(args.font_dir)
    tables = read_tables()
    for name in args.only or FIGURES:
        for theme, th in THEMES.items():
            print(save(FIGURES[name](th, tables), name, theme).relative_to(ROOT))


if __name__ == "__main__":
    main()
