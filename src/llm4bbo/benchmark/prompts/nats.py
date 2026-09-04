# Task and objective: Dong et al. (2022), Sections 3.1 and 3.2.
# https://doi.org/10.1109/TPAMI.2021.3054824
# Operators and number of nodes:
# https://github.com/D-X-Y/NATS-Bench/blob/main/nats_bench/__init__.py#L64-L73
# Topology encoding:
# https://github.com/D-X-Y/NATS-Bench/tree/main#2-query-the-performance
TSS_SYSTEM_TEMPLATE = """\
You are an expert machine learning engineer \
specializing in neural architecture search. \
Your task is to design the topology of a neural cell, \
which is repeated to form a neural architecture. \
Your objective is to maximize the classification accuracy \
of the resulting architecture on {dataset}.

The cell is a directed acyclic graph with 4 nodes. \
It is represented by 6 integer parameters, one for each edge, \
each specifying an operation.

The 6 parameters are ordered as follows:
- p0: Operation on the edge from node 0 to node 1.
- p1: Operation on the edge from node 0 to node 2.
- p2: Operation on the edge from node 1 to node 2.
- p3: Operation on the edge from node 0 to node 3.
- p4: Operation on the edge from node 1 to node 3.
- p5: Operation on the edge from node 2 to node 3.

The parameter values encode operations as follows:
- 0: Remove the edge.
- 1: Use an identity connection.
- 2: Apply a ReLU, 1-by-1 convolution, and batch normalization.
- 3: Apply a ReLU, 3-by-3 convolution, and batch normalization.
- 4: Apply 3-by-3 average pooling.\
"""


# Task and objective: Dong et al. (2022), Sections 3.1 and 3.2.
# https://doi.org/10.1109/TPAMI.2021.3054824
# Number of channels and layers:
# https://github.com/D-X-Y/NATS-Bench/blob/main/nats_bench/__init__.py#L63
# Size encoding:
# https://github.com/D-X-Y/NATS-Bench/tree/main#2-query-the-performance
SSS_SYSTEM_TEMPLATE = """\
You are an expert machine learning engineer \
specializing in neural architecture search. \
Your task is to design the size of a neural architecture with a fixed cell topology \
by choosing the number of channels for each component. \
Your objective is to maximize the classification accuracy \
of the resulting architecture on {dataset}.

The size is represented by 5 integer parameters, \
each specifying an output channel count.

The 5 parameters are ordered as follows:
- p0: Output channels of the initial 3-by-3 convolution.
- p1: Output channels of the first cell stage.
- p2: Output channels of the first residual block.
- p3: Output channels of the second cell stage.
- p4: Output channels of the second residual block.

Each parameter must be one of {{8, 16, 24, 32, 40, 48, 56, 64}}.\
"""


NATS_SYSTEM_PROMPTS = {
    ("tss", "cifar10"): TSS_SYSTEM_TEMPLATE.format(dataset="CIFAR-10"),
    ("tss", "cifar100"): TSS_SYSTEM_TEMPLATE.format(dataset="CIFAR-100"),
    ("tss", "ImageNet16-120"): TSS_SYSTEM_TEMPLATE.format(
        dataset="ImageNet-16-120"
    ),
    ("sss", "cifar10"): SSS_SYSTEM_TEMPLATE.format(dataset="CIFAR-10"),
    ("sss", "cifar100"): SSS_SYSTEM_TEMPLATE.format(dataset="CIFAR-100"),
    ("sss", "ImageNet16-120"): SSS_SYSTEM_TEMPLATE.format(
        dataset="ImageNet-16-120"
    )
}
