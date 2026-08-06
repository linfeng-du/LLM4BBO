import ast
import re
from collections.abc import Callable
from functools import partial

import numpy as np
from transformers.pipelines.text_generation import ChatType

from . import prompts


def create_prompt_fn(task_name: str) -> Callable:
    match task_name:
        case "TFBind8-Exact-v0":
            task_prompt = prompts.TFBIND_TASK.format(length=8, factor="SIX6_REF_R1")
            reference_prompt = prompts.TFBIND_REFERENCE
            stringify_fn = _tfbind_stringify_fn
        case "TFBind10-Exact-v0":
            task_prompt = prompts.TFBIND_TASK.format(length=10, factor="Pho4")
            reference_prompt = prompts.TFBIND_REFERENCE
            stringify_fn = _tfbind_stringify_fn
        case "AntMorphology-Exact-v0":
            task_prompt = prompts.ANT_MORPHOLOGY_TASK
            reference_prompt = prompts.MORPHOLOGY_REFERENCE
            stringify_fn = _morphology_stringify_fn
        case "DKittyMorphology-Exact-v0":
            task_prompt = prompts.DKITTY_MORPHOLOGY_TASK
            reference_prompt = prompts.MORPHOLOGY_REFERENCE
            stringify_fn = _morphology_stringify_fn
        case _:
            raise ValueError(f"Invalid task: {task_name}")

    def prompt_fn(
        x_ref: np.ndarray,
        y_ref: np.ndarray,
        use_tools: bool,
        x_resp: np.ndarray | None = None,
        generate_trace: bool = False
    ) -> ChatType | tuple[ChatType, ChatType]:
        if use_tools:
            tool_prompt = f"\n\n{prompts.TOOL_USE}"
        else:
            tool_prompt = ""

        if generate_trace:
            assert x_resp is not None
            final_prompt = prompts.GENERATE_TRACE.format(response=stringify_fn(x_resp))
        else:
            final_prompt = prompts.GENERATE_DESIGN

        system_prompt = f"{task_prompt}{tool_prompt}\n\n{final_prompt}"
        user_prompt = reference_prompt.format(
            references="\n".join(
                stringify_fn(x, y) for x, y in zip(x_ref, y_ref, strict=True)
            )
        )
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if x_resp is not None:
            completion = [{"role": "assistant", "content": stringify_fn(x_resp)}]
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
    x_str = f"<design>{[round(p.item(), 3) for p in x]}</design>"

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
            parsed = ast.literal_eval(matches[-1])
        except Exception:
            # Handle <design>ACGT</design>
            parsed = list(matches[-1])

        if isinstance(parsed, str):
            # Handle <design>'ACGT'</design>
            parsed = list(parsed)

        if (
            not isinstance(parsed, list)
            or len(parsed) != sequence_length
            or not all(p in BASES for p in parsed)
        ):
            return [0] * sequence_length

        return [BASES.index(p) for p in parsed]

    return np.array([parse_completion(c) for c in completions])


def _morphology_parse_fn(completions: list[str], num_parameters: int) -> np.ndarray:
    def parse_completion(completion: str) -> list[float]:
        matches = DESIGN_PATTERN.findall(completion)

        if not matches:
            return [0.0] * num_parameters

        try:
            parsed = ast.literal_eval(matches[-1])
        except Exception:
            return [0.0] * num_parameters

        if not isinstance(parsed, list) or len(parsed) != num_parameters:
            return [0.0] * num_parameters

        try:
            parsed = [float(p) for p in parsed]
        except Exception:
            return [0.0] * num_parameters

        return parsed

    return np.array([parse_completion(c) for c in completions])
