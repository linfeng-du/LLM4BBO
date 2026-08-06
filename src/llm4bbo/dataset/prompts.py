TOOL_USE = """\
You may propose 0 to 3 intermediate designs. \
When you need to verify the score of a design, \
use the `predict_score` tool, \
which returns its predicted score and uncertainty, \
then you continue reasoning.\
"""


GENERATE_TRACE = """\
Generate a concise reasoning trace that naturally leads to the target design below. \
The target design is known to be better than all provided examples.

Target design:
{response}

Do not mention or imply that the target design was provided in advance. \
Present it as the result of your own analysis. \
The target design may appear naturally during the reasoning process, \
including as an intermediate design or in a tool call. \
When giving the final answer, \
output exactly the target design and nothing else.\
"""


GENERATE_DESIGN = """\
Think step-by-step but concisely. \
Do not repeat information already provided to you. \
When giving your final answer, \
you must wrap the design within <design></design> XML tags. \
Answer only with the design and nothing else.\
"""


# TFBind8-Exact-v0 and TFBind10-Exact-v0
TFBIND_TASK = """\
You are an expert in DNA sequence design. \
Your task is to generate a new length-{length} DNA sequence, \
composed of A, C, G, and T, \
that maximizes the binding score for the transcription factor {factor}.\
"""


TFBIND_REFERENCE = """\
You are provided with example DNA sequences and their binding scores:

{references}

Design a new DNA sequence with a higher binding score than all given examples.\
"""


# AntMorphology-Exact-v0
# https://github.com/brandontrabucco/morphing-agents/tree/master/morphing_agents/mujoco/ant
ANT_MORPHOLOGY_TASK = """\
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
p12, p13, p14: Lengths of the hip, thigh, and ankle links.\
"""


# DKittyMorphology-Exact-v0
# https://github.com/brandontrabucco/morphing-agents/tree/master/morphing_agents/mujoco/dkitty
DKITTY_MORPHOLOGY_TASK = """\
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
p12, p13: Lengths of the thigh and ankle links.\
"""


MORPHOLOGY_REFERENCE = """\
You are provided with example robot morphologies and their performance scores:

{references}

Design a new robot morphology with a higher performance score than all given examples.\
"""
