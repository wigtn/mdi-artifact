"""Tests for env grid expansion and env_id hashing (FR-001 mechanics)."""

from typing import Any, Dict, List

import pytest

from mdi.grid import env_id, expand_grid, prompt_family_field


def test_env_id_stable_across_key_order_permutations() -> None:
    """Same env config -> same env_id regardless of key insertion order."""
    # Given: the same env config with keys inserted in different orders
    config_a: Dict[str, Any] = {
        "judge": "m1",
        "task": "summarization",
        "scale": "likert5",
        "temperature": 0.0,
    }
    config_b: Dict[str, Any] = {
        "temperature": 0.0,
        "scale": "likert5",
        "judge": "m1",
        "task": "summarization",
    }
    # When: both are hashed
    # Then: the env ids are identical
    assert env_id(config_a) == env_id(config_b)


def test_env_id_changes_when_any_axis_value_changes() -> None:
    """Mutating any single axis value must change the env_id."""
    # Given: a base env config (including verbatim template text per ADR-009)
    base: Dict[str, Any] = {
        "judge": "m1",
        "prompt_template": "Score the summary from 1 to 5.",
        "scale": "likert5",
        "task": "summarization",
        "temperature": 0.0,
    }
    base_id = env_id(base)
    # When: each axis value is mutated one at a time
    variant_ids: List[str] = []
    for key in base:
        mutated = dict(base)
        mutated[key] = 0.7 if key == "temperature" else f"{base[key]}_changed"
        variant_ids.append(env_id(mutated))
    # Then: every variant differs from base and from every other variant
    assert all(variant != base_id for variant in variant_ids)
    assert len(set(variant_ids)) == len(variant_ids)


def test_env_id_deterministic_across_runs() -> None:
    """env_id matches a precomputed constant — guards cross-process determinism."""
    # Given: a fixed env config
    config: Dict[str, Any] = {
        "judge": "m1",
        "scale": "likert5",
        "task": "summarization",
        "temperature": 0.0,
    }
    # When: hashed in this process (constant below was computed independently)
    # Then: format is e_<12 hex> and the value never drifts between runs
    assert env_id(config) == "e_fd1a711370c0"
    assert env_id(config) == env_id(dict(config))


def test_expand_grid_cell_count_matches_axis_product() -> None:
    """Cell count equals the product of axis lengths, with no duplicate cells."""
    # Given: axes of lengths 3, 2, 2
    axes: Dict[str, List[Any]] = {
        "judge": ["m1", "m2", "m3"],
        "scale": ["likert5", "likert10"],
        "temperature": [0.0, 0.7],
    }
    # When: the grid is expanded
    cells = expand_grid(axes)
    # Then: 3 * 2 * 2 = 12 unique cells
    assert len(cells) == 12
    assert len({env_id(cell) for cell in cells}) == 12


def test_prompt_family_field_keeps_every_template_source_text_sorted() -> None:
    """ADR-009 §3: every paraphrase's template SOURCE TEXT is in the hash input, sorted."""
    # Given: a family declared out of paraphrase-id order
    variants = [("pp_1", "rubric one"), ("pp_0", "rubric zero")]
    # When: the family field is canonicalized
    field = prompt_family_field(variants)
    # Then: sorted [paraphrase_id, template] pairs, with the verbatim template text
    assert field == [["pp_0", "rubric zero"], ["pp_1", "rubric one"]]


def test_prompt_family_field_and_env_id_are_independent_of_declaration_order() -> None:
    """A family's env_id must not depend on the order paraphrases are declared."""
    # Given: the same three paraphrases in two declaration orders
    variants_a = [("pp_2", "text-2"), ("pp_0", "text-0"), ("pp_1", "text-1")]
    variants_b = [("pp_0", "text-0"), ("pp_1", "text-1"), ("pp_2", "text-2")]
    # When: each becomes the prompt_family of an otherwise identical env
    env_a: Dict[str, Any] = {"scale": "likert5", "prompt_family": prompt_family_field(variants_a)}
    env_b: Dict[str, Any] = {"scale": "likert5", "prompt_family": prompt_family_field(variants_b)}
    # Then: identical family field and identical env_id
    assert prompt_family_field(variants_a) == prompt_family_field(variants_b)
    assert env_id(env_a) == env_id(env_b)


def test_prompt_family_field_rejects_duplicate_paraphrase_ids() -> None:
    """Two paraphrases with the same id would collide inside the family — reject it."""
    # Given: a family with a repeated paraphrase_id
    variants = [("pp_0", "a"), ("pp_0", "b")]
    # When / Then: canonicalization refuses rather than silently dropping one
    with pytest.raises(ValueError, match="duplicate paraphrase_id"):
        prompt_family_field(variants)


def test_expand_grid_order_deterministic_regardless_of_input_order() -> None:
    """Expansion order is sorted and independent of input key/value ordering."""
    # Given: the same axes declared with different key and value orders
    axes_a: Dict[str, List[Any]] = {"judge": ["m2", "m1"], "temperature": [0.7, 0.0]}
    axes_b: Dict[str, List[Any]] = {"temperature": [0.0, 0.7], "judge": ["m1", "m2"]}
    # When: both are expanded
    cells_a = expand_grid(axes_a)
    cells_b = expand_grid(axes_b)
    # Then: identical cell lists in identical (sorted) order
    assert cells_a == cells_b
    assert cells_a[0] == {"judge": "m1", "temperature": 0.0}
    assert len(cells_a) == 4
