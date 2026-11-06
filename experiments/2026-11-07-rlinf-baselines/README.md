# RLinf GPU and wafer baseline replay

This directory adds a source-referenced, colocated RLinf timing simulation to the existing rl-xy repository. It does not run native RLinf, execute numerical training, or measure real GPU/wafer performance. RLinf source reference: commit `0338bb5d85adbf0114e0b3201ea55fe2d6263ca6`; the partition algorithm in `src/rlinf_balance.py` is derived from that project's Apache-2.0 code, with the license included here.

## Inputs and results

The bundled inputs contain only test1 length metadata (192 prompts × 8 answers per model) and the measured reward-service duration for each selected answer. They do not include source code, captured program output, local paths, credentials or model weights. `inputs/manifest.json` preserves the source full-file hashes as well as hashes of the reduced files; the reduced bytes necessarily have different hashes. Prior results were produced before this packaging change and are not misrepresented as having been rerun from this directory.

| Qwen3 | GPU veRL | GPU RLinf | Wafer veRL | Wafer RLinf |
|---|---:|---:|---:|---:|
| 4B | 174.679 s | 142.452 s | 200.843 s | 195.114 s |
| 8B | 225.805 s | 183.913 s | 249.276 s | 238.990 s |
| 14B | 308.166 s | 256.711 s | 319.444 s | 295.638 s |

The prior GPU and wafer RLinf runs used the same inputs and policy per model; wafer uses 54 allocated/48 active dies and a 64 GB/die model, while GPU uses 48 A100 and 80 GB/device. These are not equal-cost hardware configurations. The historical final three-mechanism results use a different cost/CPU-reward contract and must not be divided by these baselines as a validated speedup. See `comparison-historical.json` for result identities and explicit scope.

Colocation groups all 8 samples per prompt on one rollout instance and submits rewards when its instance completes; training includes old/reference logprob, advantage, backward, one optimizer update and publication. Optional RLinf source-derived length balancing acts after advantage calculation on observed completed samples. Reference CPU swap, network, 25 GB/s host transfers, sleep/wake and control overhead are modeled; no native channel schedule, cross-TP IPC mapping, physical wafer host I/O, end-to-end hardware throughput or learning-quality calibration is implied. `arrival_rr` is one simulated possible assignment, not a verified default RLinf dispatch.

The 8B and 14B wafer training estimates are 63.796 and 63.947 GB of 64 GB including the chosen 3 GiB reserve. Their small remaining margins are not evidence that a physical device will fit.

## Reproduce

From the repository root, install the existing f2 dependencies (`python3 -m pip install -e '.[f2]'`). Runs need no GPU; wafer cost cache misses may take a long time. Each run requires a fresh `--run-id` and `--output-root`. The first example reproduces the 4B GPU policy; run IDs may be changed. The fixed seed inputs are test1 only.

```sh
python3 experiments/2026-11-07-rlinf-baselines/test_release.py
python3 experiments/2026-11-07-rlinf-baselines/run_rlinf.py \
  --platform gpu --output-root /tmp/rlinf-replay --model 4B --window test1 \
  --train-tp 2 --rollout-tp 4 --train-mb 4 --logprob-mb 4 --cap 128 \
  --assignment group_chunk --release instance --reassembly group_barrier \
  --reward-service parallel --state-mode resident_ipc --control-overhead 0.5 \
  --training-assignment arrival_rr --reference-swap cpu_model \
  --dp-balance source --balance-control-overhead 0.5 --run-id gpu-4b
```

For the 4B wafer case change `--platform gpu` to `--platform wafer` and use a new run ID. For 8B use the same TP/MB/cap but `--training-assignment input` and add `--runtime-reserve-gib 3` on wafer. For 14B use `--train-tp 4 --rollout-tp 8 --train-mb 4 --logprob-mb 16 --training-assignment arrival_rr` and add `--runtime-reserve-gib 3` on wafer. Results include full execution identity, sample/event history and status in the requested output directory; the cost cache remains outside the repository.

Wafer veRL baseline, preserving the global reward barrier and 3 GiB reserve:

```sh
python3 experiments/2026-11-07-rlinf-baselines/run_verl.py \
  --platform wafer --output-root /tmp/rlinf-replay --model 4B --window test1 \
  --mode verl --assignment striped --release global --reward-service parallel \
  --runtime-reserve-gib 3 --run-id wafer-verl-4b
```

Provider assets, operator implementation, and tests in the existing main repository remain authoritative. Cache keys bind source/hardware/mapping identity; differences in the base revision or source identity can cause the historical timing to differ and require explicit revalidation. No cache or ephemeral process logs are checked in.
