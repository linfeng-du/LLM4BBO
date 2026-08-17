import llm4bbo.patches

import fire

from datasets import Dataset

from transformers import AutoTokenizer
from trl.generation.vllm_client import VLLMClient

from vllm import LLM, SamplingParams

from llm4bbo.dataset import build_dataset
from llm4bbo.generate import GenerateWithTools
from llm4bbo.gpr import create_tool


def check_dataset(
    task_name: str = "TFBind8-Exact-v0",
    stage: str = "trace",
    use_tools: bool = True
) -> None:
    dataset = _get_dataset(task_name, stage, use_tools)
    example = dataset[0]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    prompt = tokenizer.apply_chat_template(
        dataset[0]["prompt"],
        [create_tool(task_name)] if use_tools else None,
        add_generation_prompt=True,
        tokenize=False
    )

    print(dataset[0])
    print(prompt)


def generate_colocate(
    task_name: str = "TFBind8-Exact-v0",
    stage: str = "trace",
    use_tools: bool = True,
    model: str = "Qwen/Qwen3-4B",
    thinking_budget: int = 512,
    answer_budget: int = 1024,
    max_tool_calling_iterations: int = 3
) -> None:
    dataset = _get_dataset(task_name, stage, use_tools)

    llm = LLM(model, gpu_memory_utilization=0.85)
    tokenizer = llm.get_tokenizer()

    llm.generate = GenerateWithTools(
        llm.generate,
        tokenizer,
        [create_tool(task_name)],
        thinking_budget,
        answer_budget,
        max_tool_calling_iterations
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=thinking_budget + answer_budget,
        logprobs=0
    )

    requests = llm.generate(
        [dataset[0]["prompt"]],
        [dataset[0]["chat_template_kwargs"]],
        sampling_params
    )
    print(requests[0].outputs[0].text)


def generate_server(
    host: str,
    task_name: str = "TFBind8-Exact-v0",
    stage: str = "trace",
    model: str = "Qwen/Qwen3-4B",
    thinking_budget: int = 4096,
    answer_budget: int = 1024,
    max_tool_calling_iterations: int = 3
) -> None:
    dataset = _get_dataset(task_name, stage, use_tools=True)

    client = VLLMClient(host=host)
    tokenizer = AutoTokenizer.from_pretrained(model)

    client.generate = GenerateWithTools(
        client.generate,
        tokenizer,
        [create_tool(task_name)],
        thinking_budget,
        answer_budget,
        max_tool_calling_iterations
    )

    kwargs = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": thinking_budget + answer_budget,
        "logprobs": 0
    }

    outputs = client.generate(
        [dataset[0]["prompt"]],
        [dataset[0]["chat_template_kwargs"]],
        **kwargs
    )
    print(tokenizer.decode(outputs["completion_ids"][0]))


def _get_dataset(task_name: str, stage: str, use_tools: bool) -> Dataset:
    if stage in {"trace", "sft", "offline_rl"}:
        kwargs = {
            "response_ratio": 0.5,
            "candidate_strategy": "similarity",
            "num_candidates": 20,
            "num_permutations": 2,
            "num_shots": 10
        }

        if stage == "offline_rl":
            kwargs["scale_reward"] = True

    elif stage == "online_rl":
        kwargs = {
            "dataset_size": 1000,
            "num_shots": 10
        }

    kwargs["use_tools"] = use_tools
    datasets = build_dataset(
        task_name,
        stage,
        num_designs=500,
        val_ratio=0.2,
        seed=42,
        **kwargs
    )
    return datasets["train"]


if __name__ == "__main__":
    fire.Fire()
