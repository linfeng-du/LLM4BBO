import ast
import re
from collections.abc import Callable
from functools import partial

import numpy as np
from transformers.pipelines.text_generation import ChatType


def create_prompt_fn(
    task_name: str
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray | None],
    ChatType | tuple[ChatType, ChatType]
]:
    system_prompt, user_prompt, stringify_fn = PROMPT_FN_REGISTRY[task_name]

    def prompt_fn(
        x_reference: np.ndarray,
        y_reference: np.ndarray,
        x_response: np.ndarray | None = None
    ) -> ChatType | tuple[ChatType, ChatType]:
        references = "\n".join(
            stringify_fn(x, y)
            for x, y in zip(x_reference, y_reference, strict=True)
        )
        prompt = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt.format(references=references)
            }
        ]

        if x_response is not None:
            completion = [
                {"role": "assistant", "content": stringify_fn(x_response)}
            ]
            return prompt, completion

        return prompt

    return prompt_fn


def create_parse_fn(task_name: str) -> Callable[[list[str]], np.ndarray]:
    return PARSE_FN_REGISTRY[task_name]


DESIGN_PATTERN = re.compile(r"<design>(.*?)</design>", re.DOTALL)


# TFBind8-Exact-v0 and TFBind10-Exact-v0
TFBIND_SYSTEM_PROMPT = """\
You are an expert in DNA sequence design. \
Your task is to generate a new length-{length} DNA sequence, \
composed of A, C, G, and T, \
that maximizes the binding score for the transcription factor {factor}.

Think step-by-step but concisely. \
After thinking, immediately give your final answer without any other text. \
Wrap your final answer in <design>...</design>.\
"""


TFBIND_USER_PROMPT = """\
You are provided with example DNA sequences and their binding scores:

{references}

Design a new DNA sequence with a higher binding score than all given examples.\
"""


BASES = ["A", "C", "G", "T"]


def _tfbind_stringify_fn(x: np.ndarray, y: np.ndarray | None = None) -> str:
    x_str = f"<design>{[BASES[b] for b in x]}</design>"

    if y is None:
        return x_str

    return f"DNA: {x_str}, Binding Score: {y.item()}"


def _tfbind_parse_fn(
    completions: list[str],
    sequence_length: int
) -> np.ndarray:
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


# AntMorphology-Exact-v0
# https://github.com/brandontrabucco/morphing-agents/tree/master/morphing_agents/mujoco/ant
ANT_MORPHOLOGY_SYSTEM_PROMPT = """\
You are an expert in quadruped robot morphology design. \
Your task is to generate a new morphology for the Ant quadruped robot \
that maximizes its running speed. \
The morphology is represented by 60 continuous parameters, \
grouped into 4 legs with 15 parameters per leg. \
Each leg is a 3-link kinematic chain with hip, thigh, and ankle joints. \
All parameters must be rounded to 3 decimal places.

Parameter schema (repeats for each leg):
p0, p1, p2: 3D location on the torso where the leg is mounted.
p3, p4, p5: Fixed orientation of the leg relative to the torso.
p6, p7: Midpoint and half-range of the hip joint's motion range.
p8, p9: Midpoint and half-range of the thigh joint's motion range.
p10, p11: Midpoint and half-range of the ankle joint's motion range.
p12, p13, p14: Lengths of the hip, thigh, and ankle links.

Think step-by-step but concisely. \
After thinking, immediately give your final answer without any other text. \
Wrap your final answer in <design>...</design>.\
"""


# DKittyMorphology-Exact-v0
# https://github.com/brandontrabucco/morphing-agents/tree/master/morphing_agents/mujoco/dkitty
DKITTY_MORPHOLOGY_SYSTEM_PROMPT = """\
You are an expert in quadruped robot morphology design. \
Your task is to generate a new morphology for the D'Kitty quadruped robot \
that maximizes its ability to navigate to a fixed location. \
The morphology is represented by 56 continuous parameters, \
grouped into 4 legs with 14 parameters per leg. \
Each leg is a 3-link kinematic chain with hip, thigh, and ankle joints. \
All parameters must be rounded to 3 decimal places.

Parameter schema (repeats for each leg):
p0, p1, p2: 3D location on the torso where the leg is mounted.
p3, p4, p5: Fixed orientation of the leg relative to the torso.
p6, p7: Midpoint and half-range of the hip joint's motion range.
p8, p9: Midpoint and half-range of the thigh joint's motion range.
p10, p11: Midpoint and half-range of the ankle joint's motion range.
p12, p13: Lengths of the thigh and ankle links.

Think step-by-step but concisely. \
After thinking, immediately give your final answer without any other text. \
Wrap your final answer in <design>...</design>.\
"""


MORPHOLOGY_USER_PROMPT = """\
You are provided with example robot morphologies and their performance scores:

{references}

Design a new robot morphology \
with a higher performance score than all given examples.\
"""


def _morphology_stringify_fn(
    x: np.ndarray,
    y: np.ndarray | None = None
) -> str:
    x_str = f"<design>{[round(p.item(), ndigits=3) for p in x]}</design>"

    if y is None:
        return x_str

    return f"Robot Morphology: {x_str}, Performance Score: {y.item()}"


def _morphology_parse_fn(
    completions: list[str],
    num_parameters: int
) -> np.ndarray:
    def parse_completion(completion: str) -> list[float]:
        matches = DESIGN_PATTERN.findall(completion)

        if not matches:
            return [0.0] * num_parameters

        try:
            parameters = ast.literal_eval(matches[-1])
        except Exception:
            return [0.0] * num_parameters

        if (
            not isinstance(parameters, list)
            or len(parameters) != num_parameters
        ):
            return [0.0] * num_parameters

        try:
            parameters = [float(p) for p in parameters]
        except Exception:
            return [0.0] * num_parameters

        return parameters

    return np.array([parse_completion(c) for c in completions])


PROMPT_FN_REGISTRY = {
    "TFBind8-Exact-v0": (
        TFBIND_SYSTEM_PROMPT.format(length=8, factor="SIX6_REF_R1"),
        TFBIND_USER_PROMPT,
        _tfbind_stringify_fn
    ),
    "TFBind10-Exact-v0": (
        TFBIND_SYSTEM_PROMPT.format(length=10, factor="Pho4"),
        TFBIND_USER_PROMPT,
        _tfbind_stringify_fn
    ),
    "AntMorphology-Exact-v0": (
        ANT_MORPHOLOGY_SYSTEM_PROMPT,
        MORPHOLOGY_USER_PROMPT,
        _morphology_stringify_fn
    ),
    "DKittyMorphology-Exact-v0": (
        DKITTY_MORPHOLOGY_SYSTEM_PROMPT,
        MORPHOLOGY_USER_PROMPT,
        _morphology_stringify_fn
    )
}


PARSE_FN_REGISTRY = {
    "TFBind8-Exact-v0": partial(_tfbind_parse_fn, sequence_length=8),
    "TFBind10-Exact-v0": partial(_tfbind_parse_fn, sequence_length=10),
    "AntMorphology-Exact-v0": partial(_morphology_parse_fn, num_parameters=60),
    "DKittyMorphology-Exact-v0": partial(
        _morphology_parse_fn, num_parameters=56
    )
}
