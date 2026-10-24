#!/usr/bin/env python3
"""Report historical multi-mode draft matrices with explicit unknown handling.

Usage: python -m scripts.matrix_report results/draft_matrix_*.guard_strong.jsonl
Current paper uses scripts.build_paper_evidence instead. This helper preserves
sample IDs, rejects duplicates and mismatches, and corrects baseline contrasts
within each input file. Automatic labels are not human-confirmed success.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from scripts.analyze_pairs import analyze_records, holm_adjust, outcome_summary


def load(path):
    groups = defaultdict(dict)
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get('prompt'), str):
                raise ValueError(f'{path}: string prompt required')
            sample = row.get('sample_idx', 0)
            if not isinstance(sample, int) or isinstance(sample, bool) or sample < 0:
                raise ValueError(f'{path}: invalid sample_idx')
            key = (str(row['id']), sample)
            mode = row['mode']
            if key in groups[mode]:
                raise ValueError(f'{path}: duplicate {mode} {key}')
            if 'guard_strong' not in row:
                row['guard_strong'] = row.get('guard', {'safety': 'missing'})
            groups[mode][key] = row
    if not groups:
        raise ValueError(f'{path}: no records')
    return groups


def mcnemar(base, cell):
    return analyze_records(base, cell, 'guard', 'wide')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    for path in args.paths:
        groups = load(path)
        print(path, '(QwenGuard wide; full-n positive bounds; mean per-answer acceptance)')
        for mode, rows in groups.items():
            result = outcome_summary(rows, 'guard', 'wide')
            accept = [row['accept_rate'] for row in rows.values() if row.get('accept_rate') is not None]
            mean = sum(accept)/len(accept) if accept else None
            print(mode, json.dumps({**result, 'mean_accept_rate': mean}, allow_nan=False))
        if 'baseline' not in groups:
            continue
        pairs = [{'mode': mode, **mcnemar(groups['baseline'], rows)}
                 for mode, rows in groups.items() if mode != 'baseline']
        known = [i for i, row in enumerate(pairs) if row['mcnemar_p'] is not None]
        for i, adjusted in zip(known, holm_adjust([pairs[i]['mcnemar_p'] for i in known])):
            pairs[i]['holm_p'] = adjusted
        for row in pairs:
            print('paired_vs_baseline', json.dumps(row, allow_nan=False))


if __name__ == '__main__':
    main()
