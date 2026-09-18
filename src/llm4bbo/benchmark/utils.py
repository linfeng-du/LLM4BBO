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
    categories: list[str],
    dtype: np.dtype
) -> tuple[np.ndarray | None, str | None]:
    matches = _DESIGN_PATTERN.findall(completion)

    if not matches:
        return None, "No complete <design>...</design> block found"

    try:
        design = ast.literal_eval(matches[-1].strip())
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

    if not isinstance(design, list):
        return None, f"Expected a list of parameters, got {type(design).__name__}"

    if len(design) != design_dim:
        return None, f"Expected {design_dim} parameters, got {len(design)}"

    x = []

    for index, parameter in enumerate(design):
        if parameter not in categories:
            return None, (
                f"Parameter at index {index} must be one of {categories}, "
                f"got {parameter}"
            )

        x.append(categories.index(parameter))

    try:
        with np.errstate(over="raise"):
            x = np.array(x, dtype=dtype)
    except Exception as e:
        return None, f"Could not convert the parameters to dtype {dtype}: {e}"

    return x, None


# Supported formats:
# <design>[0.1, -2.5, 3.0]</design>
# <design>['0.1', '-2.5', '3e-2']</design>
def parse_numerical(
    completion: str,
    design_dim: int,
    allowed_values: list[int] | None,
    dtype: np.dtype
) -> tuple[np.ndarray | None, str | None]:
    matches = _DESIGN_PATTERN.findall(completion)

    if not matches:
        return None, "No complete <design>...</design> block found"

    try:
        design = ast.literal_eval(matches[-1].strip())
    except Exception:
        return None, "Could not parse the design. Expected type: list[float]"

    if not isinstance(design, list):
        return None, f"Expected a list of parameters, got {type(design).__name__}"

    if len(design) != design_dim:
        return None, f"Expected {design_dim} parameters, got {len(design)}"

    x = []

    for index, parameter in enumerate(design):
        try:
            x.append(float(parameter))
        except Exception:
            return None, (
                f"Parameter at index {index} cannot be converted to float type"
            )

    if allowed_values is not None:
        for index, parameter in enumerate(x):
            if parameter not in allowed_values:
                return None, (
                    f"Parameter at index {index} must be one of {allowed_values}, "
                    f"got {parameter}"
                )

    try:
        with np.errstate(over="raise"):
            x = np.array(x, dtype=dtype)
    except Exception as e:
        return None, f"Could not convert the parameters to dtype {dtype}: {e}"

    nonfinite_indices = np.flatnonzero(~np.isfinite(x)).tolist()

    if nonfinite_indices:
        return None, (
            "All parameters must be finite; "
            f"got NaN or infinity at indices {nonfinite_indices}"
        )

    return x, None
