import ast
import re
from collections.abc import Callable
from functools import partial

import numpy as np
from transformers.pipelines.text_generation import ChatType

from . import prompt


def create_prompt_fn(task_name: str, enable_tools: bool) -> (
    Callable[
        [np.ndarray, np.ndarray, np.ndarray | None],
        ChatType | tuple[ChatType, ChatType]
    ]
):
    match task_name:
        case "TFBind8-Exact-v0":
            system_prompt = prompt.TFBIND_TASK.format(length=8, factor="SIX6_REF_R1")
            user_prompt = prompt.TFBIND_REFERENCES
            stringify_fn = _tfbind_stringify_fn
        case "TFBind10-Exact-v0":
            system_prompt = prompt.TFBIND_TASK.format(length=10, factor="Pho4")
            user_prompt = prompt.TFBIND_REFERENCES
            stringify_fn = _tfbind_stringify_fn
        case "AntMorphology-Exact-v0":
            system_prompt = prompt.ANT_MORPHOLOGY_TASK
            user_prompt = prompt.MORPHOLOGY_REFERENCES
            stringify_fn = _morphology_stringify_fn
        case "DKittyMorphology-Exact-v0":
            system_prompt = prompt.DKITTY_MORPHOLOGY_TASK
            user_prompt = prompt.MORPHOLOGY_REFERENCES
            stringify_fn = _morphology_stringify_fn
        case _:
            raise ValueError(f"Invalid task: {task_name}")

    if enable_tools:
        system_prompt += f"\n\n{prompt.TOOL_USE}"

    system_prompt += f"\n\n{prompt.FINAL_ANSWER}"

    def prompt_fn(
        x_reference: np.ndarray,
        y_reference: np.ndarray,
        x_response: np.ndarray | None = None
    ) -> ChatType | tuple[ChatType, ChatType]:
        references = "\n".join(
            stringify_fn(x, y) for x, y in zip(x_reference, y_reference, strict=True)
        )
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(references=references)}
        ]

        if x_response is not None:
            completion = [{"role": "assistant", "content": stringify_fn(x_response)}]
            return prompt, completion

        return prompt

    return prompt_fn


BASES = ["A", "C", "G", "T"]


def _tfbind_stringify_fn(x: np.ndarray, y: np.ndarray | None = None) -> str:
    x_str = f"<design>{[BASES[b] for b in x]}</design>"

    if y is None:
        return x_str

    return f"DNA: {x_str}, Binding Score: {y.item()}"


def _morphology_stringify_fn(x: np.ndarray, y: np.ndarray | None = None) -> str:
    x_str = f"<design>{[round(p.item(), ndigits=3) for p in x]}</design>"

    if y is None:
        return x_str

    return f"Robot Morphology: {x_str}, Performance Score: {y.item()}"


def create_parse_fn(task_name: str) -> Callable[[list[str]], np.ndarray]:
    match task_name:
        case "TFBind8-Exact-v0":
            return partial(_tfbind_parse_fn, sequence_length=8)
        case "TFBind10-Exact-v0":
            return partial(_tfbind_parse_fn, sequence_length=10)
        case "AntMorphology-Exact-v0":
            return partial(_morphology_parse_fn, num_parameters=60)
        case "DKittyMorphology-Exact-v0":
            return partial(_morphology_parse_fn, num_parameters=56)
        case _:
            raise ValueError(f"Invalid task: {task_name}")


DESIGN_PATTERN = re.compile(r"<design>(.*?)</design>", re.DOTALL)


def _tfbind_parse_fn(completions: list[str], sequence_length: int) -> np.ndarray:
    def parse_completion(completion: str) -> list[int]:
        matches = DESIGN_PATTERN.findall(completion)

        if not matches:
            return [0] * sequence_length

        try:
            bases = ast.literal_eval(matches[-1])
        except Exception:
            # Handle <design>ACGT</design>
            bases = list(matches[-1])

        if isinstance(bases, str):
            # Handle <design>'ACGT'</design>
            bases = list(bases)

        if (
            not isinstance(bases, list)
            or len(bases) != sequence_length
            or not all(b in BASES for b in bases)
        ):
            return [0] * sequence_length

        return [BASES.index(b) for b in bases]

    return np.array([parse_completion(c) for c in completions])


def _morphology_parse_fn(completions: list[str], num_parameters: int) -> np.ndarray:
    def parse_completion(completion: str) -> list[float]:
        matches = DESIGN_PATTERN.findall(completion)

        if not matches:
            return [0.0] * num_parameters

        try:
            parameters = ast.literal_eval(matches[-1])
        except Exception:
            return [0.0] * num_parameters

        if not isinstance(parameters, list) or len(parameters) != num_parameters:
            return [0.0] * num_parameters

        try:
            parameters = [float(p) for p in parameters]
        except Exception:
            return [0.0] * num_parameters

        return parameters

    return np.array([parse_completion(c) for c in completions])
