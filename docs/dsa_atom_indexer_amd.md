# DSA ATOM-aligned indexer + Aiter MLA (AMD)

Use this path for **GLM-5-FP8** / DeepSeek DSA models on ROCm when you want ATOM-style FP8 indexer top-k and aiter sparse MLA kernels instead of tilelang / flashmla_kv.

## Requirements

- `SGLANG_USE_AITER=1`
- ROCm with aiter build that includes: `indexer_k_quant_and_cache`, `fp8_mqa_logits`, `deepgemm_fp8_paged_mqa_logits`, `top_k_per_row_*`, `mla_decode_fwd` with FP8 scales
- DSA `page_size=64` (auto when preshuffle is available: Triton>=3.5 or `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`)

## Example launch

```bash
export SGLANG_USE_AITER=1

python3 -m sglang.launch_server \
  --model-path zai-org/GLM-5-FP8 \
  --tensor-parallel-size 8 \
  --attention-backend dsa \
  --kv-cache-dtype fp8_e4m3 \
  --dsa-indexer-impl atom \
  --dsa-topk-backend aiter \
  --dsa-prefill-backend aiter \
  --dsa-decode-backend aiter \
  --enable-dsa-aiter-mla
```

## Flags

| Flag | Purpose |
|------|---------|
| `--dsa-indexer-impl atom` | ATOM-aligned indexer (aiter cache/MQA + `top_k_per_row`) |
| `--dsa-topk-backend aiter` | Indexer top-k via aiter (unfused; do not use with fused topk) |
| `--dsa-prefill-backend aiter` / `--dsa-decode-backend aiter` | Sparse MLA via `mla_decode_fwd` with FP8 KV scales |
| `--enable-dsa-aiter-mla` | Auto-select aiter DSA MLA backends on HIP + FP8 KV |

## Environment overrides

- `SGLANG_DSA_INDEXER_IMPL=atom`
- `SGLANG_DSA_USE_AITER_MLA=1`
- `SGLANG_DSA_INDEXER_QK_ROPE_FUSION=1` (optional; fusion hook reserved for future enablement)

## Code layout

- `python/sglang/srt/layers/attention/dsa/atom_indexer.py` — aiter indexer helpers
- `python/sglang/srt/layers/attention/dsa/indexer_triton.py` — index transform Triton (from ATOM)
- `python/sglang/srt/layers/attention/dsa/dsa_aiter_mla.py` — FP8 sparse MLA metadata + decode
