# DFlash2 on Flash FP4-Local: works at the full window, MTP keeps the throughput lead

*Reference DGX Spark, 2026-11-08 to 2026-11-10 (EST). Branch `kirmorozov/df2-worker-lane`, target `flash-local-dflash2`.*

## The promise: DFlash2 beats the MTP head on its intended target

The model card's evaluation, FP8 checkpoint on SGLang, one GB300, both methods drafting 3:

| Task | MTP head, acceptance | DFlash2, acceptance | Gain |
|---|---|---|---|
| GSM8K | 3.08 | 3.26 | +6% |
| MATH-500 | 2.93 | 3.17 | +8% |
| HumanEval | 2.52 | 2.94 | +17% |
| MBPP | 2.65 | 3.06 | +15% |
| MT-Bench | 2.50 | 2.83 | +13% |

| Concurrency | MTP head, speedup over plain decoding | DFlash2 | DFlash2 over MTP |
|---|---|---|---|
| 1 | 1.35x to 1.59x | 1.63x to 1.82x | +13% to +24% |
| 8 | 1.11x to 1.25x | 1.31x to 1.48x | +17% to +23% |
| 32 | 1.12x to 1.24x | 1.43x to 1.73x | +24% to +43% |

The lead widens with concurrency: the drafter's one parallel forward stays cheap while the MTP head's four
sequential passes grow with the batch. If those ratios held on our box (a projection, not a measurement): single
stream from MTP's 55.1 to 55.4 tok/s to roughly 62 to 69; aggregate at 6 streams from ~90 to ~105 to 110; the KV
pool gain is already measured, 1,214,065 tokens and 4.63x against 935,436 and 3.57x. This drafter, trained
against the FP8 checkpoint, does not reach these on the FP4 target: 2.60 accepted per step at 3 drafts, 7 to 20% under
MTP (night 1); 2.88 at 4 drafts, 5 to 14% under at 4 to 6 streams and level at 7 to 8 (night 2).

## What it is

The Flash FP4-Local lane (`nvidia/Qwen3.8-Flash-Next-FP4-Local` on the pinned vLLM b12x image) got a second
target: the same checkpoint drafted by `incoai/Qwen3.8-Flash-Next-DFlash2` (0.6B BF16, 5 sliding-window layers,
3 draft tokens the first night, 4 the second) instead of the checkpoint's own MTP head (4 draft tokens). Same image, unit, port and served
name; only what vLLM loads differs. Speculative decoding is lossless: the drafter changes speed, never output.

## Inside the two heads

**MTP head (flash-local).** A 49th target layer. One pass reads the last accepted token (embedding 248,320 x 2560,
norm, fc 2560 to 2560) and the target's pre-final hyper-connection state (4 streams x 2560, norm, fc per stream),
runs one full-width Qwen4Exp decoder layer (QSA attention with 24 heads, 2 KV heads and its own KV cache; 4
residual streams; a 512-expert top-10 MoE, 640 wide, NVFP4), then the final norm and the shared lm_head, and
samples one token. That token and the layer's output streams feed the next pass: four passes in a row per step,
pass k waiting on pass k-1, before the 5-token verify. Trained with the checkpoint, hence 3.1 to 3.4 of 5
accepted here.

**DFlash2 drafter (flash-local-dflash2).** Two paths. The context path runs once per token: the plugin taps the
completed outputs of target layers 4, 14, 24, 34 and 44 (the inputs of 5, 15, 25, 35, 45), averages the 4 streams
to 2560 each, the drafter's fc folds the 5 x 2560 = 12,800 into 2560 and a fused projection makes keys and values
for all 5 layers (8 KV heads x 128), stored in the drafter's own sliding-window cache (2048, 1504-token blocks,
Triton). The block path runs once per step: the anchor token plus K mask tokens (trained at 1 + 7, served at K = 3,
then 4) are embedded and go through 5 dense layers of 2560, each a 2-tap grouped conv (160 groups of 16 channels,
mixing position i with i-1) around a non-causal attention (32 heads, 8 KV; queries the block, keys the block plus the
context cache) and again around the SiLU MLP (7680 wide). The final norm and lm_head give logits on the K mask
positions; the top 16 per position go to the candidate selector, two rank-256 codebooks over the vocabulary plus a
2560 to 256 projection of the hidden state, which scores every edge between consecutive candidates (logit of the
successor plus the codebook product) and walks one path from the anchor: K drafts in one forward, whatever K.
0.6B BF16, no MoE.

**Wiring.** The target is unchanged on both sides and verifies anchor + drafts in one pass. MTP reads the bottom of
the stack and loops; DFlash2 reads five points along the stack, fires once, and keeps its own cache beside the
target's 48 groups (hence the KV-group fix). More taps, less looping.

## Getting it to boot

Five boots died in turn: the image's speculative-method whitelist, missing MTP hidden states, the drafter's KV
view split under FlashInfer (fixed with Triton attention for the drafter), a forced KV layout QSA rejects, and
finally vLLM's memory check: 35.77 GiB of KV for one 262,144-token request against 18.75 GiB available.

Root cause of the last one, read from the image's `kv_cache_utils`: the hybrid KV manager draws every group's
blocks from one pool, and a pool block is as big as the widest group's summed pages. The image gave each of the
49 target layers its own group (page 3.27 MB) but kept the drafter's 5 layers in one group (5 x 6.16 MB), so
every block of the pool was 30.8 MB while a target layer used 3.27 MB of it.

Fix (plugin module `kvgroups.py`, commits 5401d1d and 7cc87e6): one KV group per drafter layer, and the drafter's
block shrunk from 3008 to 1504 tokens so its page fits under the target's. vLLM allows per-group block sizes
(scheduler aligns to the lcm, prefix hashing to the gcd). Kill switch `DF2_DRAFT_BLOCK_SIZE=off`.

## Sixth boot: it serves, full window

| | MTP (the lane's default) | DFlash2 |
|---|---|---|
| Boot | ok | ok, kvgroups lines confirmed, groups 49 x 3008 + 5 x 1504 |
| GPU KV cache | 935,436 tokens | **1,214,065 tokens** |
| Max concurrency at 262,144 tokens | 3.57x | **4.63x** |
| Outputs | reference | correct on code and prose probes |

## Speed and acceptance, warm traffic (same box, same user, same kind of load)

Two nights under the metrics collector, 20-second intervals, warm meaning prefix hit rate 90% or better and the
first hour after boot dropped. Night 1: 3 drafts, 2026-11-09 04:07 to 2026-11-10 03:36 UTC, 2,469 warm intervals,
no restart, no error, zero preemptions. Night 2: 4 drafts (`FLASH_LOCAL_DRAFT_TOKENS=4`), container up 2026-11-10
03:46 UTC, serving from 04:02, 429 warm intervals of 1,461, no restart, no error, 4 preemptions, KV pool up to 89%
full. MTP: the engine's journal from 2026-11-08, prefix hit rate 86%. Medians per interval, KV under 50% unless
said; in parentheses how many intervals.

| Streams | MTP decode, aggregate | DFlash2, 3 drafts (night 1) | step | DFlash2, 4 drafts (night 2) | step | 4 over 3 |
|---|---|---|---|---|---|---|
| 1 | (bench: 55.1 to 55.4 greedy median) | 34.2 tok/s (52) | 67 ms | 32.1, cold only (4) | 81 ms | no warm sample |
| 3 | ~74 tok/s (4 samples) | 59.6 tok/s (177) | 125 ms | 75.7, one sample | 134 ms | no warm sample |
| 4 | ~87 tok/s (2 samples) | 68.6 tok/s (651) | 150 ms | 74.9 tok/s (7) | 174 ms | +9% |
| 5 | ~87 tok/s (16 samples) | 74.8 tok/s (226) | 166 ms | 80.5 tok/s (14) | 183 ms | +8% |
| 6 | ~90 tok/s (17 samples) | 80.5 tok/s (244) | 184 ms | 85.7 tok/s (23); 82.8 at KV 50 to 65% (72) vs 80.0 (199) | 207 ms | +4% to +6% |
| 7 | ~96 tok/s (16 samples) | 83.8 tok/s at KV 50 to 65% (77) | 208 ms | 83.4 tok/s at KV 50 to 65% (50) | 224 ms | 0% |
| 8 | | 92.0 tok/s at KV 50 to 65% (325) | 217 ms | 91.0 tok/s at KV 50 to 65% (60) | 251 ms | -1% |

| | MTP, drafting 4 | DFlash2, drafting 3 (night 1) | DFlash2, drafting 4 (night 2) |
|---|---|---|---|
| Mean acceptance length | 3.1 to 3.4 of 5 | 2.60 of 4 (up to 4.0 on predictable text) | 2.88 of 5 (2.90 to 2.97 in the warm hours) |
| Per-position acceptance | | 0.72 / 0.51 / 0.37 | 0.72 / 0.51 / 0.37 / 0.28 |
| Time to first token at 4 to 7 streams | | 1.8 to 2.2 s | 1.8 to 2.2 s |
| Peak aggregate seen | 112 tok/s | 120 tok/s | 117 tok/s |

Reading: 4 drafts accepts 2.88 of 5 against 2.60 of 4, and the fourth position accepts 0.28, exactly the geometric
decay of the first three: training at block 8 gives no bonus on later positions. The step grows 10 to 16%, so the
gain is +8 to 9% aggregate at 4 to 5 streams, +4 to 6% at 6, and nothing at 7 to 8, where verifying 5 tokens for 8
streams eats the extra token. Against MTP, DFlash2 at 4 drafts is 14% behind at 4 streams, 7% at 5, 5% at 6 (7 to
20% at 3 drafts). Both nights also show the lane's prefill stall, which no drafter touches: 7 to 8 streams with KV
at 70 to 89%, prefix hits under 50%, one 1.9 s step per 4096-token chunk, aggregate 10 tok/s for an hour at a time
(night 2 in the hours 05 to 07 and 11 to 12 UTC, night 1 from 02:00 UTC). The single-stream `./bench.sh` is still the
number that settles the headline; its first run fired into the restart window (connection refused) and the waiter
is re-armed with an engine-up check.

## Open

- Done, night 2: 4 drafts, see above. Next: 5 drafts only after a clean single-stream bench at 4. Geometric decay says position 5 accepts ~0.20, +0.2 tokens a step (3.08 of 6) against a step 8 to 10% costlier: about level at the 7 to 8 streams this box runs at night, a few percent at 4 to 6.
- Controlled single-stream `./bench.sh` (MTP reference: greedy median 55.1 to 55.4 tok/s) armed to run when the box is idle. Fills the TBD column in BENCHMARKS.md.
- Quality battery (GSM8K, HumanEval) not run on this target; lossless by construction, but the lane's numbers should be reproduced once.
- Lane tuning, not drafter: while a long prompt prefills, every decode stream gets one step per 4096-token chunk, about 1.9 s on this box (aggregate fell to 8 to 12 tok/s at 6 to 7 streams for minutes overnight, on MTP too). A smaller `--max-num-batched-tokens` (1024 to 2048) would keep streams responsive during prefill at some cost to prefill throughput; worth one measured try.
- Verdict after two nights: MTP stays the lane's default. DFlash2 at 4 drafts trails 5 to 14% in aggregate decode at 4 to 6 streams and wins on KV headroom; at 3 drafts it trailed 7 to 20% and won on step latency too. Keep `flash-local-dflash2` as a documented second target, and make 4 the target's draft default (the installer's is 3).
- Drafting more is a flag, not a retrain (corrected: an earlier version of these notes said the block was 4 by training; the drafter's config says 8). The geometric projection for 4 drafts was 2.87; measured 2.88, position 4 at 0.28: block-8 training gives no bonus on later positions, so 5 drafts projects ~3.08 and no more. The teacher mismatch is the lever for the rest of the gap.
- Housekeeping: drop the `kvdiag` diagnostics once stable, fix the cosmetic "pool bytes per block" log line.

## Fine-tuning the DFlash2 drafter for FP4-Local

What changes: the teacher. The drafter stays BF16, 0.6B, same architecture, same plugin, same KV grouping and
boot path; it is trained on hidden states captured from the FP4-Local target and on text that target
regenerated, the paper's recipe with our checkpoint as teacher, at the block of 8 it was trained at (the draft
depth, 1 to 7, is the unit's flag, not the weights'). Why it should work: the gap is 0.3 to 0.6 accepted tokens per step, the size
of a teacher mismatch, and on predictable text this drafter already accepts 3.7 to 4.0 of 4 on the FP4 target;
the MTP head has no such headroom, it ships matched to the checkpoint. A drafter back at the card's 2.8 to 3.3 at
3 drafts ties our MTP at a cheaper step; a block of 5 at the card's per-position rates gives 3.3 to 3.7.

| Path | What | Data | Compute | Time, cost (estimates) | Expected |
|---|---|---|---|---|---|
| A. Adapt on the Spark | Fine-tune incoai's weights on FP4-Local features harvested from the serving engine (the plugin already captures the drafter's input features per token) | Real traffic, 5M to 10M tokens a day; a week for a first set | The Spark, nights, engine stopped; 0.6B BF16 training needs under 20 GiB | 1 to 2 weeks wall clock, electricity | Recovers much of the 0.3 to 0.6 gap; block stays 8, depth by flag |
| B. Full recipe in the cloud | Paper recipe (~800K target-regenerated samples, 6 epochs, 3072 tokens, AdamW 6e-4 cosine, frozen target features), block 8 as trained | Regenerate ~1.5B tokens with the FP4 target | 8x B200 class | Order of 2 to 4 days end to end, low thousands of dollars | Card-level acceptance at 4 or 5 drafts, 3.3 to 3.7; clear win over MTP |

Next steps in order: wire the feature dump into the plugin behind a flag and run it under normal load; run the
single-stream bench for the MTP baseline on this box; fine-tune on the harvested set and boot the result as a
third target on the same unit; read the collector; decide on path B with those numbers. Costs and durations are
estimates to be checked at the first step, not quotes.
