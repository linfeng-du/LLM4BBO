from vllm import RequestOutput


ColocateOutput = list[RequestOutput]
ServerOutput = dict[str, list[list[int]] | list[list[float]]]
