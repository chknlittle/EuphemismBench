#!/usr/bin/env python3
"""Render the three overlay SVGs that appear inlined in the
   euphemismbench-profile-two-heretics article."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_radar

with open(ROOT / "axis_scores.json") as f:
    data = json.load(f)
m = data["models"]

base    = m["base"]
heretic = m["heretic"]
gemma   = m["gemma"]
gemma4  = m["gemma4"]
gptoss  = m["gptoss20b"]

ov_bh = render_radar.render_overlay([
    ("qwen3.5-9b · pretrain",                  "#8c5548", "#6b3d32", base),
    ("qwen3.5-9b · heretic (refusal-ablated)", "#3d6a8c", "#27496b", heretic),
])

ov_3 = render_radar.render_overlay([
    ("qwen3.5-9b-base", "#8c5548", "#6b3d32", base),
    ("gemma-2-9b",      "#6b8e6b", "#4a6b4a", gemma),
    ("gemma-4-31b",     "#7a4f88", "#553966", gemma4),
])

ov_4 = render_radar.render_overlay([
    ("qwen3.5-9b-base", "#8c5548", "#6b3d32", base),
    ("gemma-2-9b",      "#6b8e6b", "#4a6b4a", gemma),
    ("gemma-4-31b",     "#7a4f88", "#553966", gemma4),
    ("gpt-oss-20b",     "#c47a3a", "#8f5520", gptoss),
])

with open("/tmp/ov_bh.svg",    "w") as f: f.write(ov_bh)
with open("/tmp/ov_3.svg",     "w") as f: f.write(ov_3)
with open("/tmp/ov_4.svg",     "w") as f: f.write(ov_4)

print("wrote /tmp/ov_bh.svg, /tmp/ov_3.svg, /tmp/ov_4.svg")
