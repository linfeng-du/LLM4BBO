import ast
import re
from collections.abc import Callable

import numpy as np
from transformers.pipelines.text_generation import ChatType


def create_prompt_fn(
    task_name: str
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray | None],
    ChatType | tuple[ChatType, ChatType]
]:
    (
        system_prompt,
        user_prompt_prefix,
        user_prompt_suffix,
        design_stringify_fn,
        reference_stringify_fn
    ) = PROMPT_FN_REGISTRY[task_name]

    def prompt_fn(
        x_references: np.ndarray,
        y_references: np.ndarray,
        x_response: np.ndarray | None = None
    ) -> ChatType | tuple[ChatType, ChatType]:
        user_prompt = "\n".join(
            [user_prompt_prefix]
            + [
                reference_stringify_fn(x, y)
                for x, y in zip(x_references, y_references, strict=True)
            ]
            + [user_prompt_suffix]
        )
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if x_response is not None:
            completion = [{
                "role": "assistant", "content": design_stringify_fn(x_response)
            }]
            return prompt, completion

        return prompt

    return prompt_fn


def create_parse_fn(task_name: str) -> Callable[[list[str]], np.ndarray]:
    return PARSE_FN_REGISTRY[task_name]


# TFBind8-Exact-v0 and TFBind10-Exact-v0
TFBIND_USER_PROMPT_PREFIX = (
    "You are provided with an offline dataset "
    "consisting of DNA sequences and their binding affinities:"
)
TFBIND_USER_PROMPT_SUFFIX = (
    "Design a new DNA sequence with improved binding affinity "
    "compared to the given examples."
)


def _tfbind_system_prompt(length: int) -> str:
    return (
        f"Your goal is to design a new length-{length} DNA sequence, "
        "composed of A, C, G, and T, "
        "that achieves the highest binding affinity "
        "with the transcription factor SIX6 REF R1. "
        "Think step-by-step but concisely within 100 words. "
        "Wrap the final answer in <answer>...</answer> "
        "and follow the exact format shown in the examples."
    )


def _tfbind_design_stringify_fn(x: np.ndarray) -> str:
    return f"<answer>{np.array(['A', 'C', 'G', 'T'])[x].tolist()}</answer>"


def _tfbind_reference_stringify_fn(x: np.ndarray, y: np.ndarray) -> str:
    return (
        f"DNA: {_tfbind_design_stringify_fn(x)}, Binding Affinity: {y.item()}"
    )


def _create_tfbind_parse_fn(length: int) -> Callable[[list[str]], np.ndarray]:
    answer_pattern = re.compile(r"<answer>(.*)</answer>")
    base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    zeros = [0] * length

    def parse_completion(completion: str) -> list[int]:
        m = answer_pattern.search(completion)

        if not m:
            return zeros

        try:
            chars = ast.literal_eval(m.group(1))
        except (ValueError, TypeError, SyntaxError):
            return zeros

        if (
            not isinstance(chars, list)
            or len(chars) != length
            or not all(c in base_to_int for c in chars)
        ):
            return zeros

        return [base_to_int[c] for c in chars]

    def tfbind_parse_fn(completions: list[str]) -> np.ndarray:
        return np.array([parse_completion(c) for c in completions], dtype=int)

    return tfbind_parse_fn


# AntMorphology-Exact-v0 and DKittyMorphology-Exact-v0
ANTMORPHOLOGY_SYSTEM_PROMPT = (
    "Your goal is to design a new morphology of the Ant quadruped robot, "
    "composed of 60 continuous parameters, "
    "such that the robot runs as fast as possible. "
    "Think step-by-step but concisely within 100 words. "
    "Wrap the final answer in <answer>...</answer> "
    "and follow the exact format shown in the examples."
)
ANTMORPHOLOGY_USER_PROMPT_PREFIX = (
    "For each design, the morphology is represented by 60 continuous values, "
    "grouped into 4 legs with 15 parameters per leg. "
    "Each leg is parameterized by "
    "the three-dimensional position of the hip joint, "
    "the angles of the hip, thigh, and ankle joints, "
    "the center and range of each of the hip, thigh, and ankle joints, "
    "and the sizes of the hip, thigh, and ankle joints. "
    "You are provided with an offline dataset "
    "consisting of robot morphologies and their performance scores:"
)

DKITTYMORPHOLOGY_SYSTEM_PROMPT = (
    "Your goal is to design a new morphology of the D'Kitty quadruped robot, "
    "composed of 56 continuous parameters, "
    "that maximizes its locomotion performance "
    "in reaching a fixed target location. "
    "Think step-by-step but concisely within 100 words. "
    "Wrap the final answer in <answer>...</answer> "
    "and follow the exact format shown in the examples."
)
DKITTYMORPHOLOGY_USER_PROMPT_PREFIX = (
    "For each design, the morphology is represented by 56 continuous values, "
    "grouped into 4 legs with 14 parameters per leg. "
    "Each leg is parameterized by "
    "the three-dimensional position of the hip joint, "
    "the angles of the hip and knee joints, "
    "the center and range of each of the hip and knee joints, "
    "the sizes of the hip and knee joints, "
    "and the center, range, and size of the foot joint. "
    "You are provided with an offline dataset "
    "consisting of robot morphologies and their performance scores:"
)

MORPHOLOGY_USER_PROMPT_SUFFIX = (
    "Design a new robot morphology with improved performance score "
    "compared to the given examples."
)


def _morphology_design_stringify_fn(x: np.ndarray) -> str:
    return f"<answer>{[f'{param:.2f}' for param in x]}</answer>"


def _morphology_reference_stringify_fn(x: np.ndarray, y: np.ndarray) -> str:
    return (
        f"Robot Morphology: {_morphology_design_stringify_fn(x)}, "
        f"Performance Score: {y.item()}"
    )


def _create_morphology_parse_fn(
    num_parameters: int
) -> Callable[[list[str]], np.ndarray]:
    answer_pattern = re.compile(r"<answer>(.*)</answer>")
    zeros = [0] * num_parameters

    def parse_completion(completion: str) -> list[float]:
        m = answer_pattern.search(completion)

        if not m:
            return zeros

        try:
            parameters = ast.literal_eval(m.group(1))
        except (ValueError, TypeError, SyntaxError):
            return zeros

        if (
            not isinstance(parameters, list)
            or len(parameters) != num_parameters
            or not all(isinstance(p, float) for p in parameters)
        ):
            return zeros

        return parameters

    def morphology_parse_fn(completions: list[str]) -> np.ndarray:
        return np.array(
            [parse_completion(c) for c in completions], dtype=float
        )

    return morphology_parse_fn


PROMPT_FN_REGISTRY = {
    "TFBind8-Exact-v0": (
        _tfbind_system_prompt(length=8),
        TFBIND_USER_PROMPT_PREFIX,
        TFBIND_USER_PROMPT_SUFFIX,
        _tfbind_design_stringify_fn,
        _tfbind_reference_stringify_fn
    ),

    "TFBind10-Exact-v0": (
        _tfbind_system_prompt(length=10),
        TFBIND_USER_PROMPT_PREFIX,
        TFBIND_USER_PROMPT_SUFFIX,
        _tfbind_design_stringify_fn,
        _tfbind_reference_stringify_fn
    ),

    "AntMorphology-Exact-v0": (
        ANTMORPHOLOGY_SYSTEM_PROMPT,
        ANTMORPHOLOGY_USER_PROMPT_PREFIX,
        MORPHOLOGY_USER_PROMPT_SUFFIX,
        _morphology_design_stringify_fn,
        _morphology_reference_stringify_fn
    ),

    "DKittyMorphology-Exact-v0": (
        DKITTYMORPHOLOGY_SYSTEM_PROMPT,
        DKITTYMORPHOLOGY_USER_PROMPT_PREFIX,
        MORPHOLOGY_USER_PROMPT_SUFFIX,
        _morphology_design_stringify_fn,
        _morphology_reference_stringify_fn
    )
}

PARSE_FN_REGISTRY = {
    "TFBind8-Exact-v0": _create_tfbind_parse_fn(length=8),
    "TFBind10-Exact-v0": _create_tfbind_parse_fn(length=10),
    "AntMorphology-Exact-v0": _create_morphology_parse_fn(num_parameters=60),
    "DKittyMorphology-Exact-v0": _create_morphology_parse_fn(num_parameters=56)
}
