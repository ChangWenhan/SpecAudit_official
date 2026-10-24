#!/usr/bin/env python3
"""Print the current main table from the validated, versioned evidence export.

Rebuild first with python -m scripts.build_paper_evidence. This command never
calls a judge or changes frozen results. Bounds show unresolved labels.
"""
import argparse
import hashlib
import json
from scripts.build_paper_evidence import OUT, RESULTS, MAIN, MODELS, DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['gptoss', 'guard_wide', 'guard_strict'], default='gptoss')
    args = parser.parse_args()
    evidence = json.loads((OUT / 'evidence.json').read_text())
    for name, expected in evidence['source_hashes'].items():
        if hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name}: changed source; rebuild evidence before reporting')
    print(f'policy={args.policy}; confirmed positive/full n; unresolved outcomes bounded, not negative')
    print('model\tmethod\tdataset\tpositive/n\tvalid\tunresolved\tpercent bounds')
    for model, pretty in MODELS:
        for slug in MAIN:
            for ds, _, _ in DATASETS:
                c = evidence['cells']['|'.join((slug, model, ds))]['judges'][args.policy]
                lo, hi = c['rate_bounds_pct']
                print(f"{pretty}\t{slug}\t{ds}\t{c['positive']}/{c['n_total']}\t{c['n_valid']}\t{c['unresolved']}\t[{lo:.2f},{hi:.2f}]")


if __name__ == '__main__':
    main()
