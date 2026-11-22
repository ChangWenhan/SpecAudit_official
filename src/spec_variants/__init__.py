"""Adapters for public speculative-decoding proposal algorithms.

These adapters separate proposal generation from SpecAudit verification.  A
variant supplies a token chain; the same exact or lossy verifier decides what
the target server emits.  Upstream source snapshots and commit IDs live under
``third_party/speculative_variants``.
"""

from .ngram import NgramPolicy, find_suffix_match
from .lookahead import LookaheadPolicy

__all__ = ["LookaheadPolicy", "NgramPolicy", "find_suffix_match"]
