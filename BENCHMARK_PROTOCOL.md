# Reproducible benchmarking preparation

Status: protocol prepared; no new GPU experiment and no FP16 baseline executed. Original scripts/CSV reports are retained as historical artifacts. Their speedup arithmetic is valid for recorded values, but quantization alone is not established as the cause.

## Existing methodology differences

| Aspect | Existing FP16 baseline | Existing quantized script | Consequence |
|---|---|---|---|
| Images | Fresh RTSP frame grabs | Saved test_frames files | Exact scene/frame identity cannot be established. |
| Prompt | What is happening in this scene? | Same literal question | Wording matches; generation workload can still differ. |
| Revision | Unpinned model/tokenizer | Unpinned model/tokenizer | Historical revision cannot be recovered from reports alone. |
| Warm-up | One attempt, can silently skip if camera unavailable | One successful saved-image inference | Warm-up conditions differ. |
| Generation limits | Not explicitly controlled | Not explicitly controlled | Cached current wrapper ignores max_new_tokens and forwards to query defaults; historical effective settings remain unverified. |
| Inference context | No explicit inference_mode | inference_mode enabled | Potential runtime/memory confound. |
| Timing | time.time; only post-inference synchronize | perf_counter; synchronize before and after | Different boundaries and timer semantics. |
| Image decode/capture | Outside inference timing | Outside inference timing | Network input variability still changes content. |
| Memory | Post-request NVML device-wide used bytes / 1e6 (decimal MB) | PyTorch cumulative peak allocated / 2**20 (MiB labelled MB) | Different scope, statistic and units; no memory-reduction claim. |
| Power | One post-inference reading | Average of before/after readings | Neither is integrated energy; no power/energy-savings claim. |
| Outputs | Overwrites fixed baseline filenames | Overwrites fixed quantized filenames | Do not rerun these scripts against historical artifacts. |

Historical baseline 4280.8 decimal MB is about 4082.5 MiB, within a 4096 MiB GPU. Comparing it with 4096 as if the units matched is misleading. The old dashboard warning remains a separately identified limitation outside this task's visual scope.

## Prepared paired experiment procedure

1. Freeze a genuine labelled manifest (or an explicitly separately designated performance-only input set), grouping correlated scenes/cameras before splitting. Use immutable image bytes, exact question strings, sample order and identical selected split across variants. Preserve input hashes and plan.
2. Use the same immutable model code/weights revision, software environment, preprocessing, token limit, temperature, top-p, seed and question wording. Document intended dtype/quantization differences only. Explicit native query settings avoid the legacy wrapper's ignored token limits.
3. Run one model process at a time. Stop other model/dashboard workloads. Record GPU identity/driver/runtime, power/thermal conditions, CPU/device placement and available memory. Avoid swapping/offloading differences being silently attributed to quantization.
4. Use the shared inference helper, inference_mode and monotonic timing. Synchronize before start and after finish. Primary latency is encoding plus generation; exclude load, decode, file/network IO and queue wait. Store raw outputs because output length affects latency. Do not invent tokens/second without measured token counts.
5. Predeclare warm-up policy and repeats. The runner supports excluded warm-ups and one sequential pass; future repeated passes can use separate uniquely named runs with the same approved plan. Report per-run mean/median/p95/std, all failures and invalid rates. Predeclare run count; avoid selecting only favorable runs. Alternate variant order across approved paired trials to reduce temporal confounding.
6. For future comparable GPU memory measurement, use bytes internally and labelled MiB externally. Record PyTorch peak allocated and peak reserved separately, resetting peaks per request after warm-up while holding the inference lock. Include loaded weights in total footprint. Record device-wide NVML memory separately; never compare it directly with allocated memory. This instrumentation is not yet added to the evaluation runner.
7. If power/energy becomes a target, use an agreed sampler and integrate time-aligned power over a defined interval. Record unsupported sensors as unavailable. End-point readings are insufficient for energy savings. Current runner intentionally records memory/power as not measured.
8. Write new unique output directories with inputs/settings/versions/revisions and raw rows. Never rewrite baseline_results.csv, quantized_results*.csv, reports or saved images. Never compare historical measurements as if they came from this controlled protocol.

## What is executable now

tools/evaluate.py validate and plan freeze and check image/question inputs without a model. After genuine-label and experiment approval, the quantized evaluation runner can produce serial, synchronized per-request latency using pinned code, explicit generation settings and excluded warm-ups. See EVALUATION.md for exact commands and approval binding.

No FP16 backend or automatic baseline fallback has been added. Before implementing/running a paired FP16 comparison, establish memory feasibility without allocating the full model, identify an appropriate hardware/memory budget, and obtain explicit user approval. A full FP16 baseline is never an automatic step on the 4 GiB GPU.

Limitations: no new paired experiment, per-request memory sampler, power integration, statistical confidence intervals or real-GPU verification of the optional native-query experiment path in Task 2. These are reported as pending, not completed measurements.
