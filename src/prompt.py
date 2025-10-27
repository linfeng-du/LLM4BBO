import numpy as np


SYSTEM_PROMPT_TFBIND8 = (
    'Your goal is to design a new length-8 DNA sequence, composed of A, C, G, and T, '
    'that achieves the highest binding affinity with the transcription factor SIX6 REF R1.'
)

REFERENCE_PROMPT_TFBIND8 = (
    'You are provided with an offline dataset '
    'consisting of DNA and their measured binding affinities:'
)

RESPONSE_PROMPT_TFBIND8 = (
    'Design a new DNA sequence '
    'with improved binding affinity compared to the given examples.'
)


def generate_prompt_tfbind8(x_refs: np.ndarray, y_refs: np.ndarray) -> list[dict[str, str]]:
    ref_dnas = [_stringify_design_tfbind8(x_ref) for x_ref in x_refs]
    ref_affinities = y_refs.squeeze().tolist()

    user_prompt = '\n'.join(
        [REFERENCE_PROMPT_TFBIND8]
        + [
            f'DNA: {dna}, Binding Affinity: {affinity}'
            for dna, affinity in zip(ref_dnas, ref_affinities)
        ]
        + [RESPONSE_PROMPT_TFBIND8]
    )
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT_TFBIND8},
        {'role': 'user', 'content': user_prompt}
    ]


def generate_completion_tfbind8(x_resp: np.ndarray) -> list[dict[str, str]]:
    resp_dna = _stringify_design_tfbind8(x_resp)
    return [{'role': 'assistant', 'content': resp_dna}]


def _stringify_design_tfbind8(x: np.ndarray) -> str:
    x_dna_chars = np.array(['A', 'C', 'G', 'T'])[x]
    return ''.join(x_dna_chars)
