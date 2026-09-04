__all__ = ["USER_PROMPT_TEMPLATE"]


USER_PROMPT_TEMPLATE = """\
The following designs and their scores are provided for reference:

{references}

Based on the examples above, \
propose a new design expected to outperform the best one. \
For numerical parameters, \
match the precision used in the examples.\
"""
