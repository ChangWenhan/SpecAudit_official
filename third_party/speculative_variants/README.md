# Speculative-decoding source snapshots

This directory holds shallow source snapshots used by the security-variant
adapters.  The model weights are kept outside the repository.

| Variant | Upstream | Source path | Adapter |
|---|---|---|---|
| Medusa | FasterDecoding/Medusa | `medusa/` | `src/spec_variants/medusa.py` |
| EAGLE / EAGLE-3 | SafeAILab/EAGLE | `eagle/` | `src/spec_variants/eagle.py` |
| Lookahead Decoding | hao-ai-lab/LookaheadDecoding | `lookahead-decoding/` | `src/spec_variants/lookahead.py` |
| N-gram speculation | vLLM `NgramGPUSpeculator` | `vllm/` | `src/spec_variants/ngram.py` |

The adapters expose one proposal chain to the existing SpecAudit verifier.
For tree-based Medusa/EAGLE they use a linear candidate path, so the code tests
acceptance risk under the paper's verifier and does not claim to reproduce the
upstream fused tree runtime.  See `PATCHES.md` for local compatibility edits.
