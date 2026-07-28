# Qwen3-VL-30B-A3B on Intel Arc Pro B70 — What Actually Works

Running Qwen3-VL-30B-A3B under `intel/llm-scaler-vllm` on an Intel Arc Pro B70 (32 GB), serving live [Frigate](https://frigate.video) NVR captions.

**Short version: it works, but not the way the model card implies.** Every pre-quantized path is a dead end on XPU. The only route that loads is FP16 weights with online `sym_int4`.

Documented 2026-07-28. Kernel work upstream is active, so check the date before trusting the throughput numbers — they are a snapshot, not a property of the hardware.

---

## What does NOT work

This is the part that saves you days. All of these are listed as supported somewhere, and all of them fail on XPU:

| Path | Result |
|---|---|
| **AWQ** | Routes to Marlin — CUDA-only, fails on XPU |
| **GPTQ** | Routes to Marlin — CUDA-only, fails on XPU |
| **AutoRound** | MoE-loader bug |
| **MXFP4** | Listed as a supported format, but **never successfully attempted for Qwen3-VL**. Every MXFP4 reference I could find in the wild refers to InternVL, not Qwen3-VL. Treat as untested, not as working. |

AWQ and GPTQ builds fail at model load — the quantization config resolves to a Marlin kernel, which has no XPU implementation. AutoRound fails separately inside the MoE weight loader. I did not preserve the tracebacks; if you hit these and can paste yours, please open an issue so the error strings are searchable.

**Do not use upstream vLLM.** Its XPU backend handles unquantized FP16 only, which will not fit this model on a 32 GB card. You need Intel's `llm-scaler-vllm` container.

## What works

Three things in combination:

1. **Intel's `llm-scaler-vllm` container** — not upstream vLLM.
2. **FP16 weights** pulled from the standard Hugging Face repo — not a pre-quantized build.
3. **Online `sym_int4`** quantization applied at load time.

The model is quantized on the way in rather than downloaded pre-quantized. That means pulling the full FP16 weights to disk, but it is the only path that reaches a running server.

<!-- Before publishing: paste your actual docker run / compose invocation below,
     with host paths and any credentials genericized. -->

## Measured

### Memory

| Item | Size |
|---|---|
| Model resident (vision encoder included, kept at 16-bit) | **17.3 GB** |
| KV cache free | **9.3 GB** |
| Concurrency at 5,120 ctx | **~20x** |
| Card total | 32 GB |

The vision encoder stays at 16-bit — it is not quantized to int4 along with the language weights. That is already included in the 17.3 GB figure.

### Throughput

| Workload | Output tokens | Wall time |
|---|---|---|
| Prose caption | 64 | **3.9 s** |
| Structured output | 180 | **9.8 s** |

Aggregate ≈ **16 tok/s** on the prose run.

**Caveat, stated plainly:** these two runs also differ in prompt length, so you cannot cleanly fit prefill and decode from them. A prefill/decode split was **not instrumented** — that needs vLLM per-request timings, which I did not capture.

What you *can* conclude without the split: 180 tokens in 9.8 s bounds decode at **≤ 18.4 tok/s even if prefill were exactly zero**. So decode is the bottleneck regardless of how the time divides.

### Why that number looks wrong

At roughly 3B active parameters and 4-bit weights, decode reads about **1.5 GB per token**. At 16 tok/s that is **~24 GB/s of effective memory bandwidth** — single-digit percent of what this card can do.

For comparison, vLLM's own [Arc Pro B-Series announcement](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b) claims **80%+ MoE hardware efficiency** via persistent zero-gap kernels, and **2.6x end-to-end on Qwen3-30B-A3B** — the same MoE architecture and parameter count as this model.

This is nowhere near that. Working hypothesis: the online `sym_int4` path falls back to a naive gather/GEMV/scatter implementation and never reaches the optimized persistent MoE kernels. That would be consistent with the loader failures above — quantized MoE on XPU is patchy across the board, and `sym_int4` may simply be the one route that still loads rather than the one that is fast.

**This is a hypothesis, not a finding.** See open questions.

### Quality

Output quality is genuinely good. On my test frames: **12/12 person count, 12/12 animal identification.** The problem is latency, not accuracy.

---

## Frigate 0.17 integration

Several open Frigate discussions report that the OpenAI GenAI provider ignores `base_url` and calls `api.openai.com` regardless. **On 0.17, pointed at a local vLLM endpoint, it works.**

```yaml
genai:
  enabled: true
  provider: openai
  base_url: http://192.168.1.x:8011/v1
  api_key: none
  model: Qwen3-VL-30B-A3B
  objects:
    - person
```

Details that may matter if yours is not working:

- **The `/v1` suffix is present** on `base_url`.
- **`api_key: none`** — and this is the proof it is not silently falling back. A request to `api.openai.com` with a null key would fail immediately. Mine produced ~24 hours of real captions.
- **Model name must match what the server registered**, not the HF repo path. `GET /v1/models` gives the exact string.

Verified over ~24 h of continuous live operation.

---

## Open / not yet tested

Listed so nobody mistakes them for answered:

- [ ] **Text vs. VL A/B.** Run `Qwen3-30B-A3B` (text, no vision) with the same quantization on the same card and compare tok/s. Same MoE architecture, same parameter count — the only variable removed is the vision wrapper. Most informative next test: if text hits the published numbers and VL does not, the bug is localized to the VL path. If both are slow, it is the quantization path.
- [ ] **Prefill/decode split** via vLLM per-request timings.
- [ ] **MXFP4 for Qwen3-VL** — never attempted here.
- [ ] **`max_pixels` sensitivity.** Qwen3-VL uses dynamic resolution, so vision token count scales with input image size. Unknown where capping it begins degrading fine-grained detail such as identifying a held object.
- [ ] Whether the [B70 llama.cpp fixes](https://github.com/Hal9000AIML/arc-pro-b70-ubuntu-gpu-speedup-bugfixes) — which address MoE slot-init and Xe2 warptile bugs with reported 2-7x gains — indicate equivalent defects in the vLLM XPU path.

## Versions

- **Hardware:** Intel Arc Pro B70, 32 GB
- **Model:** Qwen3-VL-30B-A3B
- **Frigate:** 0.17.x

<!-- Before publishing, add: llm-scaler-vllm container tag, model revision,
     GPU compute runtime, kernel, Mesa, OS. Grab them with:
       uname -r
       cat /etc/os-release
       docker ps --format '{{.Image}}'
       dpkg -l | grep -E 'intel-(opencl|level-zero)|libze|mesa'
     Benchmarks without pinned versions age into misinformation. -->

## Contributing

Corrections welcome, especially if you get a pre-quantized path working — that would mean something changed upstream and this document needs updating rather than trusting.

## References

- [vLLM — Fast and Affordable LLM serving on Intel Arc Pro B-Series](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b)
- [intel/llm-scaler](https://github.com/intel/llm-scaler)
- [vllm-project/vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)
- [arc-pro-b70-ubuntu-gpu-speedup-bugfixes](https://github.com/Hal9000AIML/arc-pro-b70-ubuntu-gpu-speedup-bugfixes)
