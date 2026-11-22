# Third-Party Licences and Attribution

SpecAudit's own code is licensed under Apache-2.0 (see `LICENSE`).  The
following third-party components are adapted or ported.  Their licences apply
to the corresponding portions.

## Hugging Face Transformers

- Project: <https://github.com/huggingface/transformers>
- Licence: Apache-2.0
- Use: `src/verifiers.py::exact_rejection_sample` is adapted from
  `transformers.generation.utils._speculative_sampling`, specialized to
  deterministic draft proposals.
- The Apache-2.0 text is reproduced in `LICENSE`.

## Medusa

- Project: <https://github.com/FasterDecoding/Medusa>
- Licence: Apache-2.0
- Use: `src/verifiers.py::typical_acceptance` adapts the fast typical branch
  and one-token sampling logic from `medusa/model/utils.py`.
- The Apache-2.0 text is reproduced in `LICENSE`.

## EAGLE / EAGLE-3

- Project: <https://github.com/SafeAILab/EAGLE>
- Licence: Apache-2.0
- Use: the local source snapshot at commit
  `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b` supplies the official EAGLE
  model and draft-tree implementation used by `src/spec_variants/eagle.py`.
- Local compatibility changes are listed in
  `third_party/speculative_variants/PATCHES.md`.

## Lookahead Decoding

- Project: <https://github.com/hao-ai-lab/LookaheadDecoding>
- Licence: Apache-2.0
- Use: the local source snapshot at commit
  `eed010da9c7b1867912675d480feec5629e0c2d0` supplies the official
  n-gram-seeded lookahead/Jacobi algorithm used by
  `src/spec_variants/lookahead.py`.

## vLLM N-gram speculator

- Project: <https://github.com/vllm-project/vllm>
- Licence: Apache-2.0
- Use: the local source snapshot at commit
  `e72fc9ae50c221180c4b8f4a94c5009714a4b199` supplies the official
  `NgramGPUSpeculator` suffix-selection contract used by
  `src/spec_variants/ngram.py`.

## FlipAttack

- Project: <https://github.com/yueliu1999/FlipAttack>
- Licence: MIT
- Use: `scripts/run_baseline.py` is a self-contained port of the prompt
  transformation and single-query evaluation loop (commit `c53880b`).
- The upstream repository and assets are not vendored.

MIT licence text:

```text
MIT License

Copyright (c) 2026 yueliu1999

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Tree of Attacks with Pruning (TAP)

- Project: <https://github.com/RICommunity/TAP>
- Licence: MIT
- Use: `scripts/run_tap.py` is a self-contained port of the attacker/evaluator
  prompts and the limited-budget search loop (commit `7bcdad3`).
- The upstream repository and data are not vendored.

MIT licence text:

```text
MIT License

Copyright (c) 2023 Robust Intelligence Community

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Datasets and model weights

The datasets and model weights referenced by the code are third-party
artifacts under their own licences and terms.  They are not redistributed in
this repository.  See `docs/DATA.md`.
