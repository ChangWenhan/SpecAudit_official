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
