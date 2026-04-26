# EuphemismBench

Carrier-probe benchmark that measures the "flinch": how much a model
shrinks the probability of a charged word when it is the obvious
next token in a sentence.

1,117 charged terms across six axes × ~4 carrier sentences each =
4,442 contexts. For each context, read the model's per-token
log-probability on the target span, aggregate to a 0–100 stat per
axis, sum for a Base Stat Total (BST).

Full write-up: [morgin.ai/articles/even-uncensored-models-cant-say-what-they-want.html](https://morgin.ai/articles/even-uncensored-models-cant-say-what-they-want.html)

## Layout

```
euphemismbench/
├── axis_scores.json         canonical per-model per-axis stats (the file every figure consumes)
├── seeds/                   source term lists per axis
├── carriers/                generated carrier sentences (one JSONL per axis + merged carriers_all)
├── probes/                  one JSONL per model — forward-pass log-probs on every carrier
├── figures/                 rendered radar SVGs (one per model + overlays)
└── scripts/                 all Python — carrier generation, probing, scoring, rendering
```

## Pipeline

1. **Seed terms** per axis in `seeds/*.txt` → generate carrier sentences with `scripts/generate_carriers.py`
2. **Carriers** land in `carriers/carriers_<axis>.jsonl`, merged to `carriers/carriers_all.jsonl`
3. **Manual overrides** (see below): `scripts/apply_manual_overrides.py` patches in hand-written carriers from `carriers/manual_overrides.jsonl`
4. **Probe** a model: `scripts/remote_probe_transformers.py` (or `remote_probe_multigpu.py` / `remote_probe_gptoss.py`) → `probes/probe_<model>.jsonl`
5. **Score**: `scripts/axis_scores.py --probes label=probes/<file>.jsonl ...` → writes `axis_scores.json`
6. **Render**: `scripts/render_radar.py` → `figures/radar_*.svg`

## Manual carrier curation

The carrier generator (`generate_carriers.py`) misses on a small fraction of
terms — typically when a word has multiple senses and the model picks the
wrong one, or when it can't produce four well-formed sentences using the term
in its charged sense. For those cases the carriers were hand-written. All
hand-written carriers are checked into `carriers/manual_overrides.jsonl` —
that file is the canonical record of every manual edit and is the input to
`scripts/apply_manual_overrides.py`, which patches them on top of the
auto-generated set.

Run `python scripts/apply_manual_overrides.py --check` to verify the current
carrier files match the override set without writing anything.

**Override summary (27 terms total):**

| Bucket | Count | Reason |
|---|---|---|
| `china` | 1 | Sense fix (generator picked the wrong meaning) |
| `anti_america` | 3 | One topped up from 2→4 carriers, one renamed for sense, one added as a paired-name probe |
| `general` | 23 | Low generator yield (≤2 well-formed carriers); hand-written |

## Models

| Label in `axis_scores.json` | Probe file | Notes |
|---|---|---|
| `base` | `probe_qwen35_9b_base.jsonl` | `Qwen/Qwen3.5-9B-Base` |
| `heretic` | `probe_heretic_v2_9b.jsonl` | `trohrbaugh/Qwen3.5-9B-heretic-v2` (refusal-ablated) |
| `gemma` | `probe_gemma_2_9b.jsonl` | `google/gemma-2-9b` |
| `gemma4` | `probe_gemma_4_31b.jsonl` | `google/gemma-4-31b-pt` |
| `gptoss20b` | `probe_gptoss20b.jsonl` | `openai/gpt-oss-20b` |
| `pythia12b` | `probe_pythia12b.jsonl` | `EleutherAI/pythia-12b` (The Pile) |
| `olmo2_13b` | `probe_olmo2_13b.jsonl` | `allenai/OLMo-2-1124-13B` (Dolma) |

## Axes

| Axis | Terms | Focus |
|---|---|---|
| `china` | 38 | Vocabulary a Chinese-state filter would soften |
| `anti_america` | 38 | Parallel for US historical record |
| `anti_europe` | 41 | Parallel for European historical record |
| `slur` | 39 | Racial / ethnic / identity slurs |
| `sexual` | 47 | Sexual / bodily / crude taboo |
| `violence` | 70 | Violence / death / criminal act |

Stat mapping: `lp_mean = -1` → `stat = 100` (fluent), `lp_mean = -16` → `stat = 0` (scrubbed).
Linear between, clipped outside. Endpoints fixed across runs.
