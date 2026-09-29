"""Environment grid expansion and env_id hashing (FR-001 mechanics).

Axis *values* (which judges, tasks, scales, ...) are config-driven and stay
pending the kickoff ADRs (ADR-002/003/006); this module only implements the
deterministic mechanics: canonical config normalization, env_id derivation,
and cartesian grid expansion.

Per ADR-009, the env_id hash input must include the prompt template *source
text* (verbatim content, not a reference id). Callers are responsible for
placing the rendered template content into the env config mapping passed to
:func:`env_id`.

Exp 3 (procedural noise floor omega^2) needs several prompt paraphrases to share
*one* env_id so the between-paraphrase variance is measurable within a single
environment (ADR-011 D1). :func:`prompt_family_field` canonicalizes the whole
paraphrase family into the env_id hash input — every paraphrase's template
source text is still hashed (ADR-009 §3 provenance preserved), but the family
collapses to a single env_id instead of one env_id per paraphrase.
"""

import hashlib
import itertools
import json
from typing import Any, Dict, List, Sequence, Set, Tuple

ENV_ID_PREFIX = "e_"
ENV_ID_HEX_LEN = 12


def canonical_json(config: Dict[str, Any]) -> str:
    """Serialize *config* to canonical JSON: sorted keys, compact separators, ASCII-safe."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def env_id(env_config: Dict[str, Any]) -> str:
    """Derive the deterministic environment id for one grid cell.

    Canonical JSON (sorted keys) -> SHA-256 -> ``e_<first 12 hex chars>``.
    Stable across key-insertion order and across processes.
    """
    digest = hashlib.sha256(canonical_json(env_config).encode("utf-8")).hexdigest()
    return f"{ENV_ID_PREFIX}{digest[:ENV_ID_HEX_LEN]}"


def prompt_family_field(variants: Sequence[Tuple[str, str]]) -> List[List[str]]:
    """Canonical ``prompt_family`` hash input: sorted ``[paraphrase_id, template]`` pairs.

    *variants* is one ``(paraphrase_id, template_source_text)`` pair per paraphrase
    in the family, for a single scale (env cells are already scale-specific). The
    result is sorted by ``paraphrase_id`` so the field — and therefore the
    env_id — is independent of the order paraphrases are declared in the config.

    Placing this whole list in the env config keeps ADR-009 §3 provenance (every
    paraphrase's template *source text*, not a reference id, is in the hash) while
    giving all paraphrases of the family ONE shared env_id, which is what makes
    the procedural floor omega^2 measurable within one environment (ADR-011 D1).
    Duplicate paraphrase ids are rejected — they would collide inside the family.
    """
    seen: Set[str] = set()
    pairs: List[List[str]] = []
    for paraphrase_id, template in variants:
        pid = str(paraphrase_id)
        if pid in seen:
            raise ValueError(f"duplicate paraphrase_id in prompt family: {pid!r}")
        seen.add(pid)
        pairs.append([pid, str(template)])
    pairs.sort(key=lambda pair: pair[0])
    return pairs


def _value_sort_key(value: Any) -> str:
    """Order axis values by their canonical JSON form (stable for mixed value types)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def expand_grid(axes: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Expand axis lists into the full cartesian grid of env config cells.

    Deterministic: axis names are sorted, and each axis's values are sorted by
    their canonical JSON form, so the output order is independent of the input
    ordering. The cell count equals the product of the axis lengths.
    """
    names = sorted(axes)
    value_lists = [sorted(axes[name], key=_value_sort_key) for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]
