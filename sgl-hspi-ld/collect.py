import os
import random
import logging
from copy import deepcopy
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import requests

import torch
import nltk
from nltk.corpus import words as nltk_words
from openai import OpenAI
from transformers import set_seed
import numpy as np
from tqdm import tqdm
from torch.utils.collect_env import get_pretty_env_info
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

GEMMA2_PROMPT = """<bos><start_of_turn>user
Please generate {num_random_words} random words.<end_of_turn>
<start_of_turn>model
Here are {num_random_words} random words: {random_words}"""

PHI3_PROMPT = """<|system|>
You are a friendly chatbot who always responds in the style of a pirate.<|end|>
<|user|>
Please generate {num_random_words} random words.<|end|>
<|assistant|>
Here are {num_random_words} random words: {random_words}"""

QWEN_2_5_PROMPT = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Please generate {num_random_words} random words.<|im_end|>
<|im_start|>assistant
Here are {num_random_words} random words: {random_words}"""

LLAMA_3_1_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Please generate {num_random_words} random words<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Here are {num_random_words} random words: {random_words}"""

MISTRAL_7B_IT_V_0_3_PROMPT = """<s>[INST] Please generate {num_random_words} random words.[/INST] Here are {num_random_words} random words: {random_words}"""

COHERE_AYA_EXPANSE_8B_PROMPT = """<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a friendly chatbot who always responds in the style of a pirate.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Please generate {num_random_words} random words.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Here are {num_random_words} random words: {random_words}"""

PROMPT_MAP = {
    "google/gemma-2-2b-it": GEMMA2_PROMPT,
    "google/gemma-2-9b-it": GEMMA2_PROMPT,
    "microsoft/Phi-3.5-mini-instruct": PHI3_PROMPT,
    "Qwen/Qwen2.5-1.5B-Instruct": QWEN_2_5_PROMPT,
    "Qwen/Qwen2.5-3B-Instruct": QWEN_2_5_PROMPT,
    "Qwen/Qwen2.5-7B-Instruct": QWEN_2_5_PROMPT,
    "Qwen/Qwen2.5-14B-Instruct": QWEN_2_5_PROMPT,
    "meta-llama/Llama-3.1-8B-Instruct": LLAMA_3_1_PROMPT,
    "meta-llama/Llama-3.1-70B-Instruct": LLAMA_3_1_PROMPT,
    "Meta-Llama-3.1-70B-Instruct": LLAMA_3_1_PROMPT,
    "mistralai/Mistral-7B-Instruct-v0.3": MISTRAL_7B_IT_V_0_3_PROMPT,
    "CohereForAI/aya-expanse-8b": COHERE_AYA_EXPANSE_8B_PROMPT,
}


@dataclass
class GenerationConfig:
    temperature: float = 0.0


@dataclass
class LogitsDatasetConfig:
    n_tokens: int = 8
    first_n_logits_per_token: int = 256


@dataclass
class RandomWordsDatasetConfig:
    # args for generating random prompts
    n_sequences: int = 512
    n_random_prompt_tokens: int = 3
    n_random_new_tokens: int = 16
    batch_size: int = 4

    def __post_init__(self):
        self.prompt_map: dict[str, str] = PROMPT_MAP


class RandomWordsDataset:
    def __init__(self, config: RandomWordsDatasetConfig):
        self.config = config
        self.random_words = []
        self.texts = None
        self._init_random_words()

    def _init_random_words(self):
        random_words: list[list[str]] = []
        nltk.download("popular", quiet=True)
        word_list = nltk_words.words()
        for _ in range(self.config.n_sequences):
            sampled_words = random.sample(word_list, self.config.n_random_prompt_tokens)
            random_words.append(sampled_words)
        self.random_words = random_words

    def add_prompt(self, model_name: str):
        random_texts: list[str] = []
        prompt = deepcopy(self.config.prompt_map[model_name])

        for words in self.random_words:
            text = prompt.format(
                num_random_words=self.config.n_random_prompt_tokens
                + self.config.n_random_new_tokens,
                random_words=" ".join(words),
            )
            random_texts.append(text)

        self.texts = random_texts

    def num_batches(self):
        return self.config.n_sequences // self.config.batch_size

    def get_batch(self, i):
        assert self.texts is not None, "Please call add_prompt before get_batch"
        assert i < self.num_batches(), "Index out of range"
        return self.texts[i * self.config.batch_size : (i + 1) * self.config.batch_size]


def split_fp32(x: np.ndarray) -> np.ndarray:
    """
    Split FP32 into sign, exponent, mantissa

    [bs, seq_len, n_logis] -> [bs, seq_len, n_logits, 3]
    """
    assert x.dtype == np.float32
    x = x.view(np.int32)
    sign = (x >> 31).astype(np.float32)
    exp = ((x >> 23) & 0xFF).astype(np.float32)
    mantissa = (x & 0x7FFFFF).astype(np.float32)

    stacked = np.stack([sign, exp, mantissa], axis=-1)
    return stacked


def create_tag(model_name: str, extra_tag: str) -> str:
    with open(Path.home().joinpath(".config/vllm/usage_stats.json"), "r") as f:
        lines = f.readlines()
    usage_stat_lists = [json.loads(line) for line in lines]
    usage_stat_lists = sorted(
        usage_stat_lists, key=lambda x: x["log_time"], reverse=True
    )
    usage_stats = None
    for us in usage_stat_lists:
        if "model_architecture" in us:
            usage_stats = us
            break
        else:
            continue

    gpu_count = usage_stats["gpu_count"]
    gpu_type = usage_stats["gpu_type"]
    dtype = str(usage_stats["dtype"])
    quantization = str(usage_stats["quantization"])
    kv_cache_dtype = str(usage_stats["kv_cache_dtype"])
    tensor_parallel = usage_stats["tensor_parallel_size"]

    tag = f"{model_name}_n-gpus-{gpu_count}_gpu-type-{gpu_type}_dtype-{dtype}_quant-{quantization}_kv-cache-dtype-{kv_cache_dtype}_tensor-parallel-{tensor_parallel}_{extra_tag}"
    return tag


def create_tag_sgl(url: str, extra_tag: str) -> str:
    # url = "http://localhost:30000/get_server_info"
    response = requests.get(url.replace("v1", "get_server_info"))
    server_info = response.json()
    model_name = server_info["served_model_name"].replace("/", "-")
    dtype = server_info["dtype"]
    kv_cache_dtype = server_info["kv_cache_dtype"]
    tensor_parallel = server_info["tp_size"]
    data_parallel = server_info["dp_size"]
    attn_backend = server_info["attention_backend"]
    speculative_alg = server_info["speculative_algorithm"]
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "-")
    tag = f"{model_name}/gpu-{gpu_name}_dtype-{dtype}_kv-cache-dtype-{kv_cache_dtype}_tensor-parallel-{tensor_parallel}_data-parallel-{data_parallel}_attn-backend-{attn_backend}_speculative-alg-{speculative_alg}"
    if extra_tag:
        tag += f"_{extra_tag}"
    return tag


def model_complete(
    client: OpenAI,
    model_name: str,
    batch: list[str],
    gen_cfg: GenerationConfig,
    logprob_cfg: LogitsDatasetConfig,
    seed: int,
    max_tokens: int,
):
    """
    texts: [batch_size,]
    logprobs: [batch_size, n_tokens]
    top_logprobs: [batch_size, n_tokens, n_logits]
    """
    completion = client.completions.create(
        model=model_name,
        prompt=batch,
        stream=False,
        best_of=1,
        n=1,
        logprobs=logprob_cfg.first_n_logits_per_token + 16,
        max_tokens=max_tokens,  # max_tokens includes the prompt, but n_tokens not
        seed=seed,
        temperature=gen_cfg.temperature,
    )

    texts = [c.text for c in completion.choices]

    logprobs = []
    top_logprobs = []
    for c in completion.choices:
        # log prob of the picked token
        if len(c.logprobs.token_logprobs) < logprob_cfg.n_tokens:
            raise ValueError(
                f"Not enough tokens generated ({len(c.logprobs.token_logprobs)} < {logprob_cfg.n_tokens})"
            )
        logprobs.append(c.logprobs.token_logprobs[: logprob_cfg.n_tokens])

        # log prob of the top n tokens
        top_logprobs_c = []  # [n_random_new_tokens, n_logits, 3]
        for i in range(logprob_cfg.n_tokens):
            token2logprob = c.logprobs.top_logprobs[i]
            top_logprobs_c_token_i = [c.logprobs.token_logprobs[i]] + list(
                token2logprob.values()
            )
            if len(top_logprobs_c_token_i) < logprob_cfg.first_n_logits_per_token:
                raise ValueError(
                    f"Not enough logits generated ({len(top_logprobs_c_token_i)} < {logprob_cfg.first_n_logits_per_token})"
                )
            top_logprobs_c.append(
                top_logprobs_c_token_i[: logprob_cfg.first_n_logits_per_token]
            )
        top_logprobs.append(top_logprobs_c)

    return texts, logprobs, top_logprobs


def create_logits_dataset(
    gen_cfg: GenerationConfig,
    random_words_cfg: RandomWordsDatasetConfig,
    logprob_cfg: LogitsDatasetConfig,
    save_dir: Path = None,
    extra_tag: str = "",
    # url: str = "http://localhost:8000/v1",
    url: str = "http://127.0.0.1:30000/v1",
    model_name: str = None,
    seed: int = 1,
    vllm_max_tokens: int = 768,
):
    check_args(
        gen_cfg=gen_cfg,
        random_words_cfg=random_words_cfg,
        logprob_cfg=logprob_cfg,
        vllm_max_tokens=vllm_max_tokens,
        save_dir=save_dir,
    )
    hparams = {
        "gen_cfg": asdict(gen_cfg),
        "random_words_cfg": asdict(random_words_cfg),
        "logprob_cfg": asdict(logprob_cfg),
        "save_dir": str(save_dir),
        "extra_tag": extra_tag,
        "url": url,
        "model_name": model_name,
        "seed": seed,
        "vllm_max_tokens": vllm_max_tokens,
    }

    set_seed(seed)

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", None),
        base_url=url,
    )
    model_name = client.models.list().data[0].id if model_name is None else model_name
    breakpoint()
    # tag = create_tag(model_name, extra_tag)
    try:
        tag = create_tag_sgl(url, extra_tag)
    except Exception as e:
        tag = extra_tag
    if save_dir is None:
        save_dir = Path(f"out/{tag}/seed-{seed}")
        logger.info(f"save_dir is not provided. Saving to {save_dir}")

    tag2idx = {tag: i for i, tag in enumerate([tag])}

    dataset = RandomWordsDataset(random_words_cfg)
    dataset.add_prompt(model_name)

    requests = []
    responses = []
    logprobs = []
    top_logprobs = []
    indices = []
    for i in tqdm(range(dataset.num_batches()), total=dataset.num_batches()):
        batch = dataset.get_batch(i)
        b_texts, b_logprobs, b_top_logprobs = model_complete(
            client=client,
            model_name=model_name,
            batch=batch,
            gen_cfg=gen_cfg,
            logprob_cfg=logprob_cfg,
            seed=seed,
            max_tokens=vllm_max_tokens,
        )
        requests.append(batch)
        responses.append(b_texts)
        logprobs.append(b_logprobs)
        top_logprobs.append(b_top_logprobs)
        indices.append(tag2idx[tag])

    logprobs = np.array(logprobs)  # [n_batches, batch_size, n_tokens]
    top_logprobs = np.array(top_logprobs)  # [n_batches, batch_size, n_tokens, n_logits]
    indices = np.array(indices)  # [n_batches,]

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "token_logits.npy", logprobs)
    np.save(save_dir / "idx.npy", indices)
    np.save(save_dir / "top_logits.npy", top_logprobs)

    env_info = get_pretty_env_info()
    with open(Path(save_dir).joinpath("env_info.txt"), "w") as f:
        f.write(env_info)

    with open(Path(save_dir).joinpath("tag2idx.json"), "w") as f:
        json.dump(tag2idx, f)

    with open(Path(save_dir).joinpath("hparams.json"), "w") as f:
        json.dump(hparams, f, indent=4)

    idx_to_req_res = {
        i: (req, res) for i, (req, res) in enumerate(zip(requests, responses))
    }
    with open(Path(save_dir).joinpath("req_res.json"), "w") as f:
        json.dump(idx_to_req_res, f, indent=4)


def check_args(
    gen_cfg: GenerationConfig,
    random_words_cfg: RandomWordsDatasetConfig,
    logprob_cfg: LogitsDatasetConfig,
    vllm_max_tokens: int,
    save_dir: Path,
):
    if random_words_cfg.n_random_new_tokens < logprob_cfg.n_tokens:
        logger.warning(
            f"Too few new tokens are asked to generate in the prompt ({random_words_cfg.n_random_new_tokens} < {logprob_cfg.n_tokens})"
        )
    if vllm_max_tokens < logprob_cfg.n_tokens:
        logger.warning(
            f"Too few tokens are **allowed** to generate in the VLLM ({vllm_max_tokens} < {logprob_cfg.n_tokens})"
        )

    if save_dir is not None:
        if save_dir.exists():
            raise ValueError(f"Directory {save_dir} already exists")


def upload_to_hub(
    repo_id: str,
    folder_path: str = "out",
    path_in_repo: str = None,
    token: str = None,
):
    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        token=token,
        repo_type="dataset",
        path_in_repo=path_in_repo,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    cli_map = {
        "collect": create_logits_dataset,
        "upload": upload_to_hub,
    }

    CLI(cli_map)
