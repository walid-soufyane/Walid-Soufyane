"""Generate assets/yield-curve.svg: an animated Nelson-Siegel yield curve for the profile README.

The curve morphs through five regimes (normal, steep, flat, inverted, humped), each one a
Nelson-Siegel parameter set:

    y(t) = b0 + b1 * (1 - e^{-t/tau}) / (t/tau) + b2 * ((1 - e^{-t/tau}) / (t/tau) - e^{-t/tau})

b0 is the long-run level, b1 the slope (short minus long) and b2 the curvature (the hump).
The animation uses SVG <animate> on the path and markers, so it plays inside GitHub's <img>.

    python assets/make_yield_curve.py
"""

from pathlib import Path

import numpy as np

BG, PANEL, GRID, TEXT, MUTED, ACCENT = "#0d1117", "#161b22", "#21262d", "#c9d1d9", "#8b949e", "#00D4AA"
FONT = "JetBrains Mono, SFMono-Regular, Menlo, Consolas, monospace"
W, H = 920, 320
X0, Y0, PW, PH = 70, 80, 780, 190  # plot area
T_MAX, Y_LO, Y_HI = 30.0, 0.0, 6.5  # maturity (years) and yield (%) ranges
CYCLE = 15  # seconds for one full loop

REGIMES = [  # (label, b0, b1, b2, tau)
    ("Normal", 4.6, -1.6, 0.6, 2.0),
    ("Steep", 5.2, -3.6, 0.5, 2.0),
    ("Flat", 4.1, -0.1, 0.0, 2.0),
    ("Inverted", 3.6, 1.7, -0.6, 1.6),
    ("Humped", 3.9, -0.4, 3.2, 1.4),
]
TENORS = [("2Y", 2), ("5Y", 5), ("10Y", 10), ("30Y", 30)]


def nelson_siegel(t, b0, b1, b2, tau):
    x = np.asarray(t, dtype=float) / tau
    f = (1 - np.exp(-x)) / x
    return b0 + b1 * f + b2 * (f - np.exp(-x))


def sx(t):  # square-root time axis: short end gets more room
    return X0 + np.sqrt(t / T_MAX) * PW


def sy(y):
    return Y0 + PH - (y - Y_LO) / (Y_HI - Y_LO) * PH


def main() -> None:
    t = np.linspace(0.25, T_MAX, 90)
    paths, tenor_y = [], []
    for _, b0, b1, b2, tau in REGIMES:
        ys = nelson_siegel(t, b0, b1, b2, tau)
        paths.append("M" + " L".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(t, ys)))
        tenor_y.append([sy(nelson_siegel(m, b0, b1, b2, tau)) for _, m in TENORS])

    n = len(REGIMES)
    # hold each regime, then morph: key times hold 0..h, morph h..1/n per regime
    key_times, values_idx = [], []
    for i in range(n):
        key_times += [i / n, i / n + 0.65 / n]
        values_idx += [i, i]
    key_times.append(1.0)
    values_idx.append(0)
    kt = ";".join(f"{k:.4f}" for k in key_times)
    spline = ";".join(["0 0 1 1", "0.45 0 0.25 1"] * n)

    def anim(attr, values):
        return (
            f'<animate attributeName="{attr}" dur="{CYCLE}s" repeatCount="indefinite" calcMode="spline" '
            f'keyTimes="{kt}" keySplines="{spline}" values="{";".join(values)}"/>'
        )

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="{FONT}" role="img" aria-label="Animated Nelson-Siegel yield curve moving between regimes">',
        f'<rect width="{W}" height="{H}" rx="14" fill="{BG}"/>',
        f'<rect x="12" y="12" width="{W - 24}" height="{H - 24}" rx="10" fill="{PANEL}"/>',
        f'<text x="{X0}" y="40" font-size="14" font-weight="700" fill="{TEXT}">Nelson-Siegel · Yield Curve Regimes</text>',
        f'<text x="{X0}" y="57" font-size="11" fill="{MUTED}">y(t) = β₀ + β₁·(1−e^(−t/τ))/(t/τ) + β₂·[(1−e^(−t/τ))/(t/τ) − e^(−t/τ)]</text>',
    ]
    for y in range(0, 7):
        out.append(f'<line x1="{X0}" x2="{X0 + PW}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{X0 - 10}" y="{sy(y) + 4:.1f}" text-anchor="end" font-size="10" fill="{MUTED}">{y}%</text>')
    for label, m in [("3M", 0.25), ("1Y", 1)] + TENORS:
        out.append(f'<line x1="{sx(m):.1f}" x2="{sx(m):.1f}" y1="{Y0}" y2="{Y0 + PH}" stroke="{GRID}"/>')
        out.append(f'<text x="{sx(m):.1f}" y="{Y0 + PH + 16}" text-anchor="middle" font-size="10" fill="{MUTED}">{label}</text>')
    out.append(f'<text x="{X0 + PW / 2}" y="{Y0 + PH + 32}" text-anchor="middle" font-size="10" fill="{MUTED}">maturity (√ scale) →</text>')

    # shaded area under the curve, the curve itself, and tenor markers
    area = [p + f" L{sx(T_MAX):.1f},{sy(0):.1f} L{sx(0.25):.1f},{sy(0):.1f} Z" for p in paths]
    out.append(f'<path d="{area[0]}" fill="{ACCENT}" fill-opacity="0.08">{anim("d", [area[i] for i in values_idx])}</path>')
    out.append(
        f'<path d="{paths[0]}" fill="none" stroke="{ACCENT}" stroke-width="2.5" stroke-linejoin="round">'
        f'{anim("d", [paths[i] for i in values_idx])}</path>'
    )
    for j, (label, m) in enumerate(TENORS):
        ys = [f"{tenor_y[i][j]:.1f}" for i in values_idx]
        out.append(
            f'<circle cx="{sx(m):.1f}" cy="{ys[0]}" r="4.5" fill="{BG}" stroke="#ffffff" stroke-width="2">{anim("cy", ys)}</circle>'
        )

    # regime label: one text per regime, visible while its curve is on screen
    for i, (name, b0, b1, b2, tau) in enumerate(REGIMES):
        start, end = i / n, (i + 1) / n
        k = [0, max(start - 0.02, 0), start + 0.02, end - 0.04, end - 0.01, 1]
        v = ["0", "0", "1", "1", "0", "0"]
        if i == 0:
            v = ["1", "1", "1", "1", "0", "0"]
            k = [0, 0.001, 0.002, end - 0.04, end - 0.01, 1]
        out.append(
            f'<g opacity="{v[0]}"><animate attributeName="opacity" dur="{CYCLE}s" repeatCount="indefinite" '
            f'keyTimes="{";".join(f"{x:.4f}" for x in k)}" values="{";".join(v)}"/>'
            f'<text x="{X0 + PW - 4}" y="40" text-anchor="end" font-size="14" font-weight="700" fill="{ACCENT}">{name}</text>'
            f'<text x="{X0 + PW - 4}" y="57" text-anchor="end" font-size="10" fill="{MUTED}">'
            f"β₀={b0:.1f}  β₁={b1:+.1f}  β₂={b2:+.1f}  τ={tau:.1f}</text></g>"
        )
    out.append(f'<text x="24" y="{H - 20}" font-size="9" fill="{MUTED}">illustrative parameters</text>')
    out.append("</svg>")

    path = Path(__file__).with_name("yield-curve.svg")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
