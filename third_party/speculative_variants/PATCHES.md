# Local compatibility patches

The vendored snapshot is kept for source traceability.  Local patches are
limited to compatibility with the paper environment and are listed here.

## EAGLE

`eagle/eagle/model/cnets.py`

- The official revision `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b` derives
  attention `head_dim` from `hidden_size / num_heads`.  Qwen3 checkpoints use
  an independent `head_dim` (for Qwen3-4B: 128), so the attention projection
  and output reshape now use `config.head_dim` when present.
- `src/spec_variants/upstream.py` supplies the `LossKwargs` typing helper that
  newer upstream code imports but Transformers 4.57.6 does not expose.
