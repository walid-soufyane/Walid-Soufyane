"""Generate assets/quant-lab.svg: two small quant visualisations for the profile README.

Left:  Monte Carlo simulation of geometric Brownian motion, dS = mu*S dt + sigma*S dW
       (64 paths, 1 year of daily steps) with the 5-95% band and the median path.
Right: Markowitz efficient frontier from 900 random long-only portfolios of 4
       hypothetical assets, coloured by Sharpe ratio, with the max-Sharpe portfolio.

All inputs are synthetic and seeded, so the image is reproducible:
    python assets/make_quant_lab.py
"""

from pathlib import Path

import numpy as np

BG, PANEL, GRID, TEXT, MUTED, ACCENT = "#0d1117", "#161b22", "#21262d", "#c9d1d9", "#8b949e", "#00D4AA"
FONT = "JetBrains Mono, SFMono-Regular, Menlo, Consolas, monospace"
W, H = 920, 360
PW, PH = 400, 250  # plot area of each panel
LX, RX, TOP = 50, 500, 70  # left/right panel origins, top of plot areas


def scale(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)


def polyline(xs, ys):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def gbm_panel(rng) -> list[str]:
    s0, mu, sigma, n_paths, n_steps = 100.0, 0.07, 0.20, 64, 252
    dt = 1 / n_steps
    shocks = rng.standard_normal((n_paths, n_steps))
    log_paths = np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks, axis=1)
    paths = s0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    lo, hi = 55, 175
    xs = [scale(t, 0, n_steps, LX, LX + PW) for t in range(n_steps + 1)]
    out = []
    for y_val in (60, 80, 100, 120, 140, 160):  # gridlines
        y = scale(y_val, lo, hi, TOP + PH, TOP)
        out.append(f'<line x1="{LX}" x2="{LX + PW}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{LX - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="{MUTED}">{y_val}</text>')
    p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=0)
    band_top = [scale(v, lo, hi, TOP + PH, TOP) for v in np.clip(p95, lo, hi)]
    band_bot = [scale(v, lo, hi, TOP + PH, TOP) for v in np.clip(p5, lo, hi)]
    out.append(
        f'<polygon points="{polyline(xs, band_top)} {polyline(xs[::-1], band_bot[::-1])}" fill="{ACCENT}" fill-opacity="0.10"/>'
    )
    out.append('<g clip-path="url(#clipL)">')
    for i, path in enumerate(paths):
        ys = [scale(v, lo, hi, TOP + PH, TOP) for v in np.clip(path, lo, hi)]
        out.append(
            f'<polyline class="p" style="animation-delay:{i * 0.04:.2f}s" pathLength="1" '
            f'points="{polyline(xs[::3], ys[::3])}" fill="none" stroke="{ACCENT}" stroke-opacity="0.28" stroke-width="1"/>'
        )
    ys = [scale(v, lo, hi, TOP + PH, TOP) for v in p50]
    out.append(
        f'<polyline class="p" style="animation-delay:2.8s" pathLength="1" points="{polyline(xs[::3], ys[::3])}" '
        f'fill="none" stroke="#ffffff" stroke-width="2"/>'
    )
    # histogram of S_T, drawn as horizontal bars growing from the right edge of the plot
    counts, edges = np.histogram(np.clip(paths[:, -1], lo, hi), bins=14, range=(lo, hi))
    for j, c in enumerate(counts):
        if c == 0:
            continue
        y_top = scale(edges[j + 1], lo, hi, TOP + PH, TOP)
        y_bot = scale(edges[j], lo, hi, TOP + PH, TOP)
        width = c / counts.max() * 70
        out.append(
            f'<rect class="h" style="animation-delay:{j * 0.05:.2f}s" x="{LX + PW - width:.1f}" y="{y_top + 1:.1f}" '
            f'width="{width:.1f}" height="{y_bot - y_top - 2:.1f}" fill="{ACCENT}" fill-opacity="0.35"/>'
        )
    out.append("</g>")
    out += [
        f'<text x="{LX + PW - 6}" y="{TOP + 14}" text-anchor="end" font-size="10" fill="{ACCENT}">distribution of S_T</text>',
        f'<text x="{LX}" y="{TOP - 30}" font-size="14" font-weight="700" fill="{TEXT}">Monte Carlo · Geometric Brownian Motion</text>',
        f'<text x="{LX}" y="{TOP - 13}" font-size="11" fill="{MUTED}">dS = μS dt + σS dW   μ=7%  σ=20%  64 paths · 252 steps</text>',
        f'<text x="{LX + PW / 2}" y="{TOP + PH + 22}" text-anchor="middle" font-size="10" fill="{MUTED}">trading days →</text>',
        f'<rect x="{LX + 8}" y="{TOP + PH - 44}" width="104" height="36" rx="5" fill="{PANEL}" fill-opacity="0.85"/>',
        f'<text x="{LX + 16}" y="{TOP + PH - 29}" font-size="10" fill="{ACCENT}">▬ 5–95% band</text>',
        f'<text x="{LX + 16}" y="{TOP + PH - 15}" font-size="10" fill="#ffffff">▬ median path</text>',
    ]
    return out


def frontier_panel(rng) -> list[str]:
    mu = np.array([0.04, 0.07, 0.10, 0.13])
    vol = np.array([0.06, 0.12, 0.18, 0.26])
    corr = np.array([[1, 0.2, 0.1, 0.0], [0.2, 1, 0.5, 0.3], [0.1, 0.5, 1, 0.6], [0.0, 0.3, 0.6, 1]])
    cov = corr * np.outer(vol, vol)
    rf = 0.02
    w = rng.dirichlet(np.ones(4) * 0.6, size=900)
    ret = w @ mu
    risk = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    sharpe = (ret - rf) / risk
    xlo, xhi, ylo, yhi = 0.04, 0.27, 0.035, 0.135

    def px(r):
        return scale(r, xlo, xhi, RX, RX + PW)

    def py(m):
        return scale(m, ylo, yhi, TOP + PH, TOP)

    out = []
    for v in (0.05, 0.10, 0.15, 0.20, 0.25):
        out.append(f'<line x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{TOP}" y2="{TOP + PH}" stroke="{GRID}"/>')
        out.append(f'<text x="{px(v):.1f}" y="{TOP + PH + 14}" text-anchor="middle" font-size="10" fill="{MUTED}">{v:.0%}</text>')
    for m in (0.04, 0.06, 0.08, 0.10, 0.12):
        out.append(f'<line x1="{RX}" x2="{RX + PW}" y1="{py(m):.1f}" y2="{py(m):.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{RX - 8}" y="{py(m) + 4:.1f}" text-anchor="end" font-size="10" fill="{MUTED}">{m:.0%}</text>')
    out.append('<g clip-path="url(#clipR)">')
    s_lo, s_hi = sharpe.min(), sharpe.max()
    for i, (r, m, s) in enumerate(zip(risk, ret, sharpe)):
        t = (s - s_lo) / (s_hi - s_lo)
        out.append(
            f'<circle class="d" style="animation-delay:{i * 0.0025:.3f}s" cx="{px(r):.1f}" cy="{py(m):.1f}" r="2.2" fill="{ACCENT}" fill-opacity="{0.12 + 0.75 * t:.2f}"/>'
        )
    # upper envelope of the cloud approximates the efficient frontier
    order = np.argsort(risk)
    best, pts = -1.0, []
    for i in order:
        if ret[i] > best:
            best = ret[i]
            pts.append((px(risk[i]), py(ret[i])))
    out.append(
        f'<polyline class="p" style="animation-delay:2.6s" pathLength="1" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" '
        f'fill="none" stroke="#ffffff" stroke-width="2"/>'
    )
    k = int(np.argmax(sharpe))
    cx, cy = px(risk[k]), py(ret[k])
    out += [
        f'<line x1="{px(0.04):.1f}" y1="{py(rf + sharpe[k] * 0.04):.1f}" x2="{px(0.26):.1f}" y2="{py(rf + sharpe[k] * 0.26):.1f}" '
        f'stroke="{ACCENT}" stroke-dasharray="4 4" stroke-opacity="0.7"/>',
        "</g>",
        f'<circle class="ring" cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" stroke="#ffffff" stroke-width="2"/>',
        f'<text x="{cx + 10:.1f}" y="{cy + 16:.1f}" font-size="10" fill="#ffffff">max Sharpe {sharpe[k]:.2f}</text>',
        f'<text x="{RX}" y="{TOP - 30}" font-size="14" font-weight="700" fill="{TEXT}">Markowitz · Efficient Frontier</text>',
        f'<text x="{RX}" y="{TOP - 13}" font-size="11" fill="{MUTED}">900 random portfolios · 4 assets · colour = Sharpe</text>',
        f'<text x="{RX + PW / 2}" y="{TOP + PH + 30}" text-anchor="middle" font-size="10" fill="{MUTED}">volatility →</text>',
        f'<text x="{RX + PW - 4}" y="{TOP + 14}" text-anchor="end" font-size="10" fill="{ACCENT}">- - capital market line</text>',
        f'<text x="{RX - 36}" y="{TOP + PH / 2}" text-anchor="middle" font-size="10" fill="{MUTED}" '
        f'transform="rotate(-90 {RX - 36} {TOP + PH / 2})">expected return →</text>',
    ]
    return out


def main() -> None:
    rng = np.random.default_rng(2026)
    body = gbm_panel(rng) + frontier_panel(rng)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="{FONT}" role="img" aria-label="Monte Carlo GBM paths and a Markowitz efficient frontier (simulated data)">',
        "<style>"
        ".p{stroke-dasharray:1;stroke-dashoffset:1;animation:draw 10s ease-out infinite both}"
        "@keyframes draw{0%{stroke-dashoffset:1;opacity:1}30%{stroke-dashoffset:0}80%{opacity:1}92%,100%{stroke-dashoffset:0;opacity:0}}"
        ".h{transform-box:fill-box;transform-origin:right;animation:grow 10s ease-out infinite both}"
        "@keyframes grow{0%,32%{transform:scaleX(0);opacity:1}48%{transform:scaleX(1)}80%{opacity:1}92%,100%{transform:scaleX(1);opacity:0}}"
        ".d{animation:pop 10s ease-out infinite both}"
        "@keyframes pop{0%{opacity:0}4%,80%{opacity:1}92%,100%{opacity:0}}"
        ".ring{transform-box:fill-box;transform-origin:center;animation:pulse 2s ease-in-out infinite}"
        "@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.6);opacity:.4}}"
        "@media (prefers-reduced-motion: reduce){.p,.h,.d,.ring{animation:none;stroke-dashoffset:0}}"
        "</style>",
        f'<defs><clipPath id="clipL"><rect x="{LX}" y="{TOP}" width="{PW}" height="{PH}"/></clipPath>'
        f'<clipPath id="clipR"><rect x="{RX}" y="{TOP}" width="{PW}" height="{PH}"/></clipPath></defs>',
        f'<rect width="{W}" height="{H}" rx="14" fill="{BG}"/>',
        f'<rect x="12" y="12" width="{W / 2 - 18}" height="{H - 24}" rx="10" fill="{PANEL}"/>',
        f'<rect x="{W / 2 + 6}" y="12" width="{W / 2 - 18}" height="{H - 24}" rx="10" fill="{PANEL}"/>',
        *body,
        f'<text x="{LX}" y="{H - 20}" font-size="9" fill="{MUTED}">simulated data · seed 2026</text>',
        "</svg>",
    ]
    out = Path(__file__).with_name("quant-lab.svg")
    out.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
