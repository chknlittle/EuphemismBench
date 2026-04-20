#!/usr/bin/env python3
"""Render hexagonal radar charts (Pokemon-style) as inline SVG."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AXIS_SCORES_PATH = ROOT / "axis_scores.json"
FIGURES_DIR = ROOT / "figures"

AXIS_ORDER = ["china", "anti_america", "anti_europe", "violence", "sexual", "slur"]
AXIS_LABEL = {
    "china":        "Anti-China",
    "anti_america": "Anti-America",
    "anti_europe":  "Anti-Europe",
    "slur":         "Slurs",
    "sexual":       "Sexual",
    "violence":     "Violence",
}

# Start at top, go clockwise
ANGLES_DEG = [-90, -30, 30, 90, 150, 210]


def pt(cx, cy, r, theta_deg):
    t = math.radians(theta_deg)
    return cx + r * math.cos(t), cy + r * math.sin(t)


def render(model_label, stats, accent="#4d6a79", accent_stroke="#3a5260",
           width=420, height=380, bst=None):
    cx, cy = width / 2, height / 2 + 6
    r_out = 132
    # grid rings
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'role="img" aria-label="{model_label} flinch profile">']
    # background rings (25/50/75/100)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        r = r_out * frac
        poly = []
        for deg in ANGLES_DEG:
            x, y = pt(cx, cy, r, deg)
            poly.append(f"{x:.1f},{y:.1f}")
        opacity = 0.16 if frac == 1.0 else 0.08
        out.append(f'<polygon points="{" ".join(poly)}" fill="none" '
                   f'stroke="rgba(29,25,21,{opacity})" stroke-width="1"/>')
    # scale tick labels (25, 50, 75, 100) along top axis
    for frac, label in zip([0.25, 0.5, 0.75, 1.0], ["25", "50", "75", "100"]):
        x, y = pt(cx, cy, r_out * frac, -90)
        out.append(f'<text x="{x+4:.1f}" y="{y+3:.1f}" font-family="IBM Plex Mono, monospace" '
                   f'font-size="8.5" fill="rgba(29,25,21,0.42)">{label}</text>')
    # axis spokes
    for deg in ANGLES_DEG:
        x, y = pt(cx, cy, r_out, deg)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
                   f'stroke="rgba(29,25,21,0.14)" stroke-width="1"/>')
    # polygon for model — bigger = more flinch (we invert the fluency stat)
    poly_pts = []
    for axis, deg in zip(AXIS_ORDER, ANGLES_DEG):
        v = (100 - stats[axis]["stat"]) / 100.0
        x, y = pt(cx, cy, r_out * v, deg)
        poly_pts.append((x, y))
    poly_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in poly_pts)
    out.append(f'<polygon points="{poly_str}" fill="{accent}" fill-opacity="0.22" '
               f'stroke="{accent_stroke}" stroke-width="2"/>')
    # vertex dots
    for x, y in poly_pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{accent_stroke}"/>')
    # axis labels and numeric values
    for axis, deg in zip(AXIS_ORDER, ANGLES_DEG):
        x, y = pt(cx, cy, r_out + 20, deg)
        # anchor based on angle
        if deg == -90:
            anchor, dy = "middle", 0
        elif deg == 90:
            anchor, dy = "middle", 10
        elif -90 < deg < 90:
            anchor, dy = "start", 4
        else:
            anchor, dy = "end", 4
        out.append(f'<text x="{x:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}" '
                   f'font-family="IBM Plex Mono, monospace" font-size="10" '
                   f'font-weight="600" fill="#1d1915" letter-spacing="0.05em">'
                   f'{AXIS_LABEL[axis].upper()}</text>')
        val = 100 - stats[axis]["stat"]
        out.append(f'<text x="{x:.1f}" y="{y+dy+12:.1f}" text-anchor="{anchor}" '
                   f'font-family="IBM Plex Mono, monospace" font-size="11" '
                   f'fill="{accent_stroke}">{val:.0f}</text>')
    # model label and total flinch in bottom-left
    if bst is not None:
        out.append(f'<text x="16" y="{height-22}" font-family="IBM Plex Mono, monospace" '
                   f'font-size="8.5" letter-spacing="0.12em" fill="rgba(29,25,21,0.42)">'
                   f'TOTAL FLINCH</text>')
        out.append(f'<text x="16" y="{height-8}" font-family="Playfair Display, serif" '
                   f'font-size="18" fill="#1d1915">{bst:.0f}</text>')
    out.append('</svg>')
    return "\n".join(out)


def render_overlay(models, width=520, height=440):
    """Two-model overlay on the same hex, for comparison."""
    cx, cy = width / 2, height / 2 + 6
    r_out = 150
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'role="img" aria-label="EuphemismBench flinch-profile overlay">']
    for frac in [0.25, 0.5, 0.75, 1.0]:
        r = r_out * frac
        poly = []
        for deg in ANGLES_DEG:
            x, y = pt(cx, cy, r, deg)
            poly.append(f"{x:.1f},{y:.1f}")
        opacity = 0.18 if frac == 1.0 else 0.08
        out.append(f'<polygon points="{" ".join(poly)}" fill="none" '
                   f'stroke="rgba(29,25,21,{opacity})" stroke-width="1"/>')
    for deg in ANGLES_DEG:
        x, y = pt(cx, cy, r_out, deg)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
                   f'stroke="rgba(29,25,21,0.14)" stroke-width="1"/>')
    # axis labels
    for axis, deg in zip(AXIS_ORDER, ANGLES_DEG):
        x, y = pt(cx, cy, r_out + 22, deg)
        if deg == -90:
            anchor, dy = "middle", 0
        elif deg == 90:
            anchor, dy = "middle", 10
        elif -90 < deg < 90:
            anchor, dy = "start", 4
        else:
            anchor, dy = "end", 4
        out.append(f'<text x="{x:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}" '
                   f'font-family="IBM Plex Mono, monospace" font-size="10" '
                   f'font-weight="600" fill="#1d1915" letter-spacing="0.05em">'
                   f'{AXIS_LABEL[axis].upper()}</text>')
    # each model polygon — bigger = more flinch (we invert the fluency stat)
    for label, color_fill, color_stroke, stats in models:
        poly_pts = []
        for axis, deg in zip(AXIS_ORDER, ANGLES_DEG):
            v = (100 - stats[axis]["stat"]) / 100.0
            x, y = pt(cx, cy, r_out * v, deg)
            poly_pts.append((x, y))
        poly_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in poly_pts)
        out.append(f'<polygon points="{poly_str}" fill="{color_fill}" fill-opacity="0.22" '
                   f'stroke="{color_stroke}" stroke-width="2"/>')
        for x, y in poly_pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color_stroke}"/>')
    # legend (bottom) — rows grow upward so 3-model doesn't overflow viewBox
    lx = 18
    n = len(models)
    ly = height - 8 - (n - 1) * 16
    out.append(f'<text x="{lx}" y="{ly-14}" font-family="IBM Plex Mono, monospace" '
               f'font-size="8.5" letter-spacing="0.12em" fill="rgba(29,25,21,0.42)">OVERLAY</text>')
    for i, (label, color_fill, color_stroke, _) in enumerate(models):
        y = ly + i * 16
        out.append(f'<rect x="{lx}" y="{y-9}" width="14" height="10" '
                   f'fill="{color_fill}" fill-opacity="0.28" stroke="{color_stroke}" stroke-width="1.5"/>')
        out.append(f'<text x="{lx+22}" y="{y}" font-family="IBM Plex Mono, monospace" '
                   f'font-size="11" fill="#1d1915">{label}</text>')
    out.append('</svg>')
    return "\n".join(out)


def main():
    with open(AXIS_SCORES_PATH) as f:
        data = json.load(f)
    base    = data["models"]["base"]
    heretic = data["models"]["heretic"]
    gemma   = data["models"].get("gemma")
    gemma4  = data["models"].get("gemma4")
    gptoss  = data["models"].get("gptoss20b")
    pythia  = data["models"].get("pythia12b")
    olmo2   = data["models"].get("olmo2_13b")

    # Flinch totals: invert each axis stat (100 − fluency) and sum.
    bst_b = sum(100 - base[a]["stat"]    for a in AXIS_ORDER)
    bst_h = sum(100 - heretic[a]["stat"] for a in AXIS_ORDER)

    svg_b = render("qwen3.5-9b-base", base,    accent="#8c5548", accent_stroke="#6b3d32", bst=bst_b)
    svg_h = render("heretic-v2-9b",   heretic, accent="#8f8372", accent_stroke="#6b6151", bst=bst_h)
    svg_overlay = render_overlay([
        ("qwen3.5-9b-base", "#8c5548", "#6b3d32", base),
        ("heretic-v2-9b",   "#8f8372", "#6b6151", heretic),
    ])

    with open(FIGURES_DIR / "radar_base.svg",    "w") as f: f.write(svg_b)
    with open(FIGURES_DIR / "radar_heretic.svg", "w") as f: f.write(svg_h)
    with open(FIGURES_DIR / "radar_overlay.svg", "w") as f: f.write(svg_overlay)
    msg = "wrote radar_base.svg, radar_heretic.svg, radar_overlay.svg"

    if gemma is not None:
        bst_g = sum(100 - gemma[a]["stat"] for a in AXIS_ORDER)
        svg_g = render("gemma-2-9b", gemma, accent="#6b8e6b", accent_stroke="#4a6b4a", bst=bst_g)
        with open(FIGURES_DIR / "radar_gemma.svg", "w") as f: f.write(svg_g)
        msg += ", radar_gemma.svg"

    if gemma4 is not None:
        bst_g4 = sum(100 - gemma4[a]["stat"] for a in AXIS_ORDER)
        svg_g4 = render("gemma-4-31b", gemma4, accent="#7a4f88", accent_stroke="#553966", bst=bst_g4)
        with open(FIGURES_DIR / "radar_gemma4.svg", "w") as f: f.write(svg_g4)
        msg += ", radar_gemma4.svg"

    if gptoss is not None:
        bst_go = sum(100 - gptoss[a]["stat"] for a in AXIS_ORDER)
        svg_go = render("gpt-oss-20b", gptoss, accent="#c47a3a", accent_stroke="#8f5520", bst=bst_go)
        with open(FIGURES_DIR / "radar_gptoss20b.svg", "w") as f: f.write(svg_go)
        msg += ", radar_gptoss20b.svg"

    if pythia is not None:
        bst_py = sum(100 - pythia[a]["stat"] for a in AXIS_ORDER)
        svg_py = render("pythia-12b", pythia, accent="#3f7a5a", accent_stroke="#27553d", bst=bst_py)
        with open(FIGURES_DIR / "radar_pythia12b.svg", "w") as f: f.write(svg_py)
        msg += ", radar_pythia12b.svg"

    if olmo2 is not None:
        bst_om = sum(100 - olmo2[a]["stat"] for a in AXIS_ORDER)
        svg_om = render("olmo-2-13b", olmo2, accent="#b08d4c", accent_stroke="#7a5f2a", bst=bst_om)
        with open(FIGURES_DIR / "radar_olmo2_13b.svg", "w") as f: f.write(svg_om)
        msg += ", radar_olmo2_13b.svg"

    if gemma is not None and gemma4 is not None:
        xfam_models = [
            ("qwen3.5-9b-base", "#8c5548", "#6b3d32", base),
            ("gemma-2-9b",      "#6b8e6b", "#4a6b4a", gemma),
            ("gemma-4-31b",     "#7a4f88", "#553966", gemma4),
        ]
        if gptoss is not None:
            xfam_models.append(("gpt-oss-20b", "#c47a3a", "#8f5520", gptoss))
        if pythia is not None:
            xfam_models.append(("pythia-12b", "#3f7a5a", "#27553d", pythia))
        if olmo2 is not None:
            xfam_models.append(("olmo-2-13b", "#b08d4c", "#7a5f2a", olmo2))
        svg_xfam = render_overlay(xfam_models)
        with open(FIGURES_DIR / "radar_crossfamily.svg", "w") as f: f.write(svg_xfam)
        msg += ", radar_crossfamily.svg"

    # Open-data reference overlay: Pythia + OLMo alone, to let the reader see the two
    # independent open-data pretrains on the same grid before we pile commercial labs on top.
    if pythia is not None and olmo2 is not None:
        svg_openfloor = render_overlay([
            ("pythia-12b", "#3f7a5a", "#27553d", pythia),
            ("olmo-2-13b", "#b08d4c", "#7a5f2a", olmo2),
        ])
        with open(FIGURES_DIR / "radar_openfloor.svg", "w") as f: f.write(svg_openfloor)
        msg += ", radar_openfloor.svg"
    print(msg)


if __name__ == "__main__":
    main()
