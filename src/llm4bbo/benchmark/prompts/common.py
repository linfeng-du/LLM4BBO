__all__ = [
    "DESIGN_GENERATION_PROMPT_TEMPLATE",
    "TOOL_USE_PROMPT_TEMPLATE",
    "TRAJECTORY_GENERATION_PROMPT_TEMPLATE",
    "USER_PROMPT_TEMPLATE"
]


TOOL_USE_PROMPT_TEMPLATE = """\
Before giving your final answer, \
you may call the `predict_score` tool up to {max_tool_calls} times. \
The tool accepts a design \
and returns the design's predicted score and the tool's uncertainty. \
Use this information to guide your subsequent reasoning.\
"""


TRAJECTORY_GENERATION_PROMPT_TEMPLATE = """\
Produce a complete trajectory that naturally leads to the target design below, \
which is known to outperform all provided examples:

{target}

Use the following structure for the trajectory:
- Wrap each reasoning segment in <think></think> tags.
- Wrap each tool call in <tool_call></tool_call> tags.
- Wrap each corresponding tool response in <tool_response></tool_response> tags.
- After each tool response, continue with a new <think></think> block.
- Enclose the final design in <design></design> tags.

Follow these requirements for the trajectory:
- Each reasoning segment wrapped in <think></think> tags \
must not exceed {thinking_budget} tokens.
- Do not mention or imply that the target design was provided or known in advance.
- Do not mention or imply that there is a predetermined or undisclosed target.
- Let the target design emerge naturally from the reasoning and any tool results.
- Conclude with the target design as the final answer.\
"""


DESIGN_GENERATION_PROMPT_TEMPLATE = """\
Each reasoning segment wrapped in <think></think> tags \
must not exceed {thinking_budget} tokens. \
Enclose your final answer in <design></design> tags.\
"""


USER_PROMPT_TEMPLATE = """\
The following designs and their scores are provided for reference:

{references}

Based on the examples above, \
propose a new design expected to outperform the best one. \
For numerical parameters, \
match the precision used in the examples.\
"""
