# Task and objective: Kim et al. (2023), Appendix A.1.
# https://doi.org/10.52202/075280-2958
RNA_SYSTEM_TEMPLATE = """\
You are an expert molecular biologist \
specializing in RNA-RNA binding and RNA sequence design. \
Your task is to design an RNA sequence of exactly 14 bases using only U, G, C, and A. \
Your objective is to maximize its binding score \
for the following target RNA sequence:

{target}\
"""
