__all__ = [
    "evenly_ranked_indices",
    "parse_categorical",
    "parse_numerical"
]

import ast
import re

import numpy as np


def evenly_ranked_indices(y: np.ndarray, num_designs: int) -> np.ndarray:
    ranked_indices = y.squeeze(-1).argsort()
    evenly_spaced_ranks = np.linspace(0, len(y) - 1, num_designs).round().astype(int)
    return ranked_indices[evenly_spaced_ranks]


_DESIGN_PATTERN = re.compile(r"<design>(.*?)</design>", re.DOTALL)


# Supported formats:
# <design>['A', 'C', 'G', 'T']</design>
# <design>[A, C, G, T]</design>
# <design>ACGT</design>
# <design>'ACGT'</design>
def parse_categorical(
    completion: str,
    design_dim: int,
    categories: list[str]
) -> tuple[list[int], bool]:
    matches = _DESIGN_PATTERN.findall(completion)

    if not matches:
        return [0] * design_dim, False

    try:
        design = ast.literal_eval(matches[-1])
    except Exception:
        text = matches[-1].strip()

        if text.startswith("[") and text.endswith("]"):
            # Handle cases like <design>[A, C, G, T]</design>
            design = [item.strip() for item in text[1:-1].split(",")]
        else:
            # Handle cases like <design>ACGT</design>
            design = list(text)

    if isinstance(design, str):
        # Handle cases like <design>'ACGT'</design>
        design = list(design)

    if (
        not isinstance(design, list)
        or len(design) != design_dim
        or not all(cat in categories for cat in design)
    ):
        return [0] * design_dim, False

    return [categories.index(c) for c in design], True


# Supported formats:
# <design>[0.1, -2.5, 3.0]</design>
# <design>['0.1', '-2.5', '3e-2']</design>
def parse_numerical(
    completion: str,
    design_dim: int,
    allowed_values: list[int] | None
) -> tuple[list[float], bool]:
    fallback_value = float(allowed_values[0]) if allowed_values is not None else 0.0
    matches = _DESIGN_PATTERN.findall(completion)

    if not matches:
        return [fallback_value] * design_dim, False

    try:
        design = ast.literal_eval(matches[-1])
    except Exception:
        return [fallback_value] * design_dim, False

    if not isinstance(design, list) or len(design) != design_dim:
        return [fallback_value] * design_dim, False

    try:
        design = [float(p) for p in design]
    except Exception:
        return [fallback_value] * design_dim, False

    if not all(np.isfinite(design)):
        return [fallback_value] * design_dim, False

    if allowed_values is not None and any(p not in allowed_values for p in design):
        return [fallback_value] * design_dim, False

    return design, True
