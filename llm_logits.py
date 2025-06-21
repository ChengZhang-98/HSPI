from argparse import ArgumentParser
from dataclasses import dataclass
import math
import random
from copy import deepcopy
import logging
import time
from pprint import pformat
from functools import reduce
from pathlib import Path
import yaml
import pickle
import os
import socket
import shutil
from itertools import combinations
import sys

import fire
import tqdm
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.collect_env import get_pretty_env_info
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    HqqConfig,
)
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaConfig
from huggingface_hub import HfApi
import nltk
from nltk.corpus import words as nltk_words
from optimum.quanto import QuantizedModelForCausalLM, qint8
from transformers.utils.logging import set_verbosity_error as set_hf_verbosity_error
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
import matplotlib.pyplot as plt

from blackbox_locking.logging import set_logging_verbosity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("blackbox_locking." + __name__)


def smart_time(seconds: int):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:.2f}s"
    else:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m {seconds % 60:.2f}s"


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
Please generate {num_random_words} random words<|im_end|>
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
    "mistralai/Mistral-7B-Instruct-v0.3": MISTRAL_7B_IT_V_0_3_PROMPT,
    "CohereForAI/aya-expanse-8b": COHERE_AYA_EXPANSE_8B_PROMPT,
}


@dataclass
class RandomWordsDatasetConfig:
    # args for generating random prompts
    n_sequences: int
    n_random_prompt_tokens: int
    n_random_new_tokens: int
    batch_size: int
    prompt_map: dict[str, str]


class RandomWordsDataset:
    def __init__(self, config: RandomWordsDatasetConfig):
        self.config = config
        self.random_words = []
        self.texts = None
        self.tokenized_batched_encodings = None
        self._init_random_words()

    def _init_random_words(self):
        random_words = []
        nltk.download("popular", quiet=True)
        word_list = nltk_words.words()
        for _ in range(self.config.n_sequences):
            sampled_words = random.sample(word_list, self.config.n_random_prompt_tokens)
            random_words.append(sampled_words)
        self.random_words = random_words

    def _init_texts(self, model_name: str):
        random_texts = []
        prompt = deepcopy(self.config.prompt_map[model_name])

        for words in self.random_words:
            text = prompt.format(
                num_random_words=self.config.n_random_prompt_tokens + self.config.n_random_new_tokens,
                random_words=" ".join(words),
            )
            random_texts.append(text)
        self.texts = random_texts

    def _tokenize_texts(self, model_name: str, tokenizer):
        encodings = []

        for i in range(self.config.n_sequences // self.config.batch_size):
            batch = self.texts[i * self.config.batch_size : (i + 1) * self.config.batch_size]
            encodings.append(
                tokenizer(batch, padding=True, truncation=False, return_tensors="pt", add_special_tokens=False)
            )

        self.tokenized_batched_encodings = encodings

    def prompt_and_tokenize(self, model_name: str, tokenizer):
        self._init_texts(model_name)
        self._tokenize_texts(model_name, tokenizer)

    def num_batches(self):
        return self.config.n_sequences // self.config.batch_size

    def get_batch(self, i):
        assert self.tokenized_batched_encodings is not None
        assert i < len(self.tokenized_batched_encodings)
        return self.tokenized_batched_encodings[i]

    def get_text_batch(self, i):
        assert self.texts is not None
        assert i < self.config.n_sequences // self.config.batch_size
        return self.texts[i * self.config.batch_size : (i + 1) * self.config.batch_size]


@dataclass
class LogitsDatasetConfig:
    start_token_idx: int
    n_tokens: int
    first_n_logits_per_token: int
    log_prob: bool = True


def split_fp32(x: torch.Tensor):
    """
    Split FP32 into sign, exponent, mantissa

    [bs, seq_len, n_logis] -> [bs, seq_len, n_logits, 3]
    """
    assert x.dtype == torch.float32
    x = x.view(torch.int32)
    sign = (x >> 31).float()
    exp = ((x >> 23) & 0xFF).float()
    mantissa = (x & 0x7FFFFF).float()

    stacked = torch.stack([sign, exp, mantissa], dim=-1)
    return stacked


AVAILABLE_Q_TAGS = [
    "bf16",
    "fp16",
    "fp32",
    "bnb-8bit",
    "bnb-4bit",
    "hqq-4bit",
    "quanto-wi8ai8",
    "bf16-sdpa",
    "fp16-sdpa",
    "fp32-sdpa",
    "bf16-flashattn2",
    "fp16-flashattn2",
    "fp32-flashattn2",
]


def build_llm(model_name: str, q_tag: str):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16-sdpa": torch.bfloat16,
        "fp16-sdpa": torch.float16,
        "fp32-sdpa": torch.float32,
        "bf16-flashattn2": torch.bfloat16,
        "fp16-flashattn2": torch.float16,
        "fp32-flashattn2": torch.float32,
    }
    if q_tag in ["bnb-8bit", "bnb-4bit", "hqq-4bit", "quanto-wi8ai8"]:
        if q_tag in ["bnb-8bit", "bnb-4bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=q_tag == "bnb-8bit",
                load_in_4bit=q_tag == "bnb-4bit",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, trust_remote_code=True
            )
        elif q_tag == "hqq-4bit":
            hqq_config = HqqConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False, axis=0)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cuda",
                quantization_config=hqq_config,
            )
        elif q_tag == "quanto-wi8ai8":
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
            model = QuantizedModelForCausalLM.quantize(model, weights=qint8, activations=qint8, exclude="lm_head")
        else:
            raise ValueError(f"Unsupported quantizer tag {q_tag}")
        model.eval()
    elif q_tag in [
        "bf16",
        "fp16",
        "fp32",
        "bf16-sdpa",
        "fp16-sdpa",
        "fp32-sdpa",
        "bf16-flashattn2",
        "fp16-flashattn2",
        "fp32-flashattn2",
    ]:
        model_kwargs = {"torch_dtype": dtype_map[q_tag], "trust_remote_code": True}
        if "sdpa" in q_tag:
            model_kwargs["_attn_implementation"] = "sdpa"
        elif "flashattn2" in q_tag:
            model_kwargs["_attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["_attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        model.cuda()
    else:
        raise ValueError(f"Unsupported quantizer tag {q_tag}")
    return model


@torch.no_grad()
def create_logits_dataset(
    model_names: list[str],
    q_tags: list[str],
    generation_cfg: GenerationConfig,
    random_words_dataset_cfg: RandomWordsDatasetConfig,
    logits_dataset_cfg: LogitsDatasetConfig,
):
    logits: list[np.ndarray] = []
    idx_list: list[np.ndarray] = []
    label_to_idx = {q_tag: i for i, q_tag in enumerate(q_tags)}
    device_prefix = torch.cuda.get_device_name().replace(" ", "-")

    random_words_dataset = RandomWordsDataset(random_words_dataset_cfg)
    prog_bar = tqdm.tqdm(total=len(model_names) * len(q_tags) * random_words_dataset.num_batches())

    model = None
    requests_responses: list = []
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        random_words_dataset.prompt_and_tokenize(model_name, tokenizer)
        logger.info(f"Example requests for {model_name}:\n{random_words_dataset.texts[0]}")

        for q_tag in q_tags:
            del model
            example_response_peeked = False
            torch.cuda.empty_cache()
            model = build_llm(model_name, q_tag)

            for i in range(random_words_dataset.num_batches()):
                batch = random_words_dataset.get_batch(i)
                inputs = batch.to("cuda")  # [bs, seq_len]
                prompt_len = inputs.input_ids.size(1)
                outputs = model.generate(**inputs, generation_config=generation_cfg)

                if not example_response_peeked:
                    response = tokenizer.decode(outputs.sequences[0])
                    logger.info(f"Example response for {model_name} ({q_tag}):\n{response}")
                    example_response_peeked = True

                # save requests and responses
                if len(requests_responses) < i + 1:
                    requests_responses.append({})
                    requests_responses[i]["request"] = {
                        model_name: deepcopy(random_words_dataset.get_text_batch(i)),
                    }
                    requests_responses[i]["response"] = {}
                    requests_responses[i]["response"][f"{model_name}_{device_prefix}_{q_tag}"] = tokenizer.batch_decode(
                        outputs.sequences
                    )
                else:
                    if model_name not in requests_responses[i]["request"]:
                        requests_responses[i]["request"][model_name] = deepcopy(random_words_dataset.get_text_batch(i))
                    requests_responses[i]["response"][f"{model_name}_{device_prefix}_{q_tag}"] = tokenizer.batch_decode(
                        outputs.sequences
                    )

                lgt = outputs.logits
                lgt = torch.stack(lgt, dim=1)  # [bs, seq_len, vocab_size]

                start_token_idx = logits_dataset_cfg.start_token_idx
                if start_token_idx is None:
                    start_token_idx = prompt_len
                num_tokens_per_seq = logits_dataset_cfg.n_tokens
                if num_tokens_per_seq is None:
                    num_tokens_per_seq = random_words_dataset_cfg.n_random_new_tokens

                lgt = lgt[:, :num_tokens_per_seq, : logits_dataset_cfg.first_n_logits_per_token]
                # [bs, n_tokens, first_n_logits_per_token]
                log_prob = torch.nn.functional.log_softmax(lgt, dim=-1)
                log_prob = split_fp32(log_prob)  # [bs, n_tokens, first_n_logits_per_token, 3]
                logits.append(log_prob.cpu().numpy())
                idx_list.append(label_to_idx[q_tag] * np.ones(log_prob.size(0), dtype=np.int32))
                prog_bar.update(1)

    logits = np.concatenate(logits, axis=0)  # [n_samples, n_tokens, first_n_logits_per_token, 3]
    idx_list = np.concatenate(idx_list, axis=0)  # [n_samples]

    random_words_dataset.tokenized_batched_encodings = None

    label_to_idx = {f"{device_prefix}_{q_tag}": i for i, q_tag in enumerate(q_tags)}

    return logits, idx_list, label_to_idx, requests_responses  # , unique_responses


# vocab size:
# gemma-2:                              256000
# phi3:                                  32064
# llama-3:                              128256
# mistralai/Mistral-7B-Instruct-v0.2:    32000
# qwen-2.5:                             152064


def wrapped_create_logits_dataset(
    model_names: list[str] = ["google/gemma-2b-it"],
    q_tags: list[str] = ["bf16-flashattn2", "fp16-flashattn2"],
    texts_n_seqs: int = 16,
    texts_n_random_prompt_tokens: int = 8,
    texts_n_random_new_tokens: int = 8,
    texts_batch_size: int = 1,
    logits_start_token_idx: int = None,
    logits_n_tokens: int = None,
    logits_first_n_logits_per_token: int = None,
    logits_log_prob: bool = True,
    seed: int = 1,
    output_dir: str = None,
    overwrite_output_dir: bool = False,
    hf_push_to_hub: bool = False,
    hf_token: str = None,
    hf_repo_id: str = None,
    hf_path_to_repo: str = None,
):
    """
    Feed random words to a language model and get logits for each token in the generated text.

    Args:
        model_names: List of model names to use.
        q_tags: List of quantization tags to use. Must be one of ["bf16", "fp16", "fp32", "bf16-sdpa", "fp16-sdpa", "fp32-sdpa", "bf16-flashattn2", "fp16-flashattn2", "fp32-flashattn2"].
        texts_n_seqs: Number of random sequences to feed to the model.
        texts_n_random_prompt_tokens: Number of random tokens in the prompt.
        texts_n_random_new_tokens: Number of random tokens to generate.
        texts_batch_size: Batch size for generating random words.
        logits_start_token_idx: Start token index for storing logits. Defaults to the end of the prompt.
        logits_n_tokens: Number of tokens to store logits for. Defaults to the number of new tokens generated.
        logits_first_n_logits_per_token: Number of logits to store for each token. Defaults to 30000.
        logits_log_prob: Whether to store log probabilities. Defaults to True.
        seed: Random seed. Defaults to 1.
        output_dir: Output directory to save the logits dataset.
        overwrite_output_dir: Whether to overwrite the output directory if it exists. Defaults to False.
    """
    args = deepcopy(locals())
    logger.info(f"Arguments:\n{pformat(args)}")
    if output_dir is None:
        logger.warning("No output directory specified. Results will not be saved.")
    else:
        if Path(output_dir).exists() and not overwrite_output_dir:
            raise ValueError(f"Output directory {output_dir} already exists. Use --overwrite-output-dir to overwrite.")
    args["hf_token"] = None
    transformers.set_seed(seed)

    generation_cfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=True,
        output_logits=True,
        max_new_tokens=texts_n_random_new_tokens,
        min_new_tokens=texts_n_random_new_tokens,
    )

    random_random_cfg = RandomWordsDatasetConfig(
        n_sequences=texts_n_seqs,
        n_random_prompt_tokens=texts_n_random_prompt_tokens,
        n_random_new_tokens=texts_n_random_new_tokens,
        batch_size=texts_batch_size,
        prompt_map=PROMPT_MAP,
    )

    logits_dataset_cfg = LogitsDatasetConfig(
        start_token_idx=logits_start_token_idx,
        n_tokens=logits_n_tokens,
        first_n_logits_per_token=logits_first_n_logits_per_token,
        log_prob=logits_log_prob,
    )

    logits, idx_list, label_to_idx, requests_responses = create_logits_dataset(
        model_names=model_names,
        q_tags=q_tags,
        generation_cfg=generation_cfg,
        random_words_dataset_cfg=random_random_cfg,
        logits_dataset_cfg=logits_dataset_cfg,
    )

    if output_dir is not None or hf_push_to_hub:
        remove_after_upload = False
        if output_dir is None:
            remove_after_upload = True
            output_dir = Path(f"tmp-dataset-{time.strftime('%Y%m%d-%H%M%S')}")
        output_dir = Path(output_dir)
        dataset_dir = output_dir / "logits_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        np.save(dataset_dir / "logits.npy", logits)
        np.save(dataset_dir / "idx.npy", idx_list)

        with open(dataset_dir / "label_to_idx.yaml", "w") as f:
            yaml.dump(label_to_idx, f)

        with open(dataset_dir / "requests_responses.yaml", "w") as f:
            yaml.dump(requests_responses, f)

        # with open(dataset_dir / "unique_responses.yaml", "w") as f:
        #     yaml.dump(unique_responses, f)

        with open(dataset_dir / "args.yaml", "w") as f:
            yaml.dump(args, f)

        # env info
        env_info = get_pretty_env_info()
        env_info += f"\ntransformers version: {transformers.__version__}"
        with open(Path(dataset_dir).joinpath("env_info.txt"), "w") as f:
            f.write(env_info)

        logger.info(f"📦 Logits dataset saved to {dataset_dir}")

        if hf_push_to_hub:
            assert hf_repo_id is not None, "Please provide a Hugging Face repository ID"
            if hf_path_to_repo is None:
                machine_name = socket.gethostname().replace(" ", "-").replace("/", "-")
                device_prefix = torch.cuda.get_device_name().replace(" ", "-")
                machine_device = f"{machine_name}_{device_prefix}"
                q_tags_str = "_".join(q_tags)
                model_names_esc = "_".join(model_names).replace("/", "-")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                hf_path_to_repo = (
                    f"{machine_device}/seed-{seed}/{model_names_esc}/{q_tags_str}/{timestamp}/logits_dataset"
                )

            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo_id, private=True, exist_ok=True, repo_type="dataset")
            api.upload_folder(
                repo_id=hf_repo_id, folder_path=dataset_dir, path_in_repo=hf_path_to_repo, repo_type="dataset"
            )
            logger.info(f"☁️ Logits dataset uploaded to Hugging Face Hub: {hf_repo_id}/{hf_path_to_repo}")
        if remove_after_upload:
            logger.info(f"Removing temporary directory {output_dir}")
            shutil.rmtree(output_dir)


def load_logit_datasets(
    logits_datasets: list[str],
    truncate_first_n_tokens: int = None,
    truncate_first_n_logits: int = None,
    q_tags_to_remove: list[str] = None,
    q_tags_to_keep: list[str] = None,
):
    if (q_tags_to_remove is not None) and (q_tags_to_keep is not None):
        raise ValueError("Cannot specify both q_tags_to_remove and q_tags_to_keep")
    if len(set(logits_datasets)) != len(logits_datasets):
        # find duplicates
        seen = set()
        duplicates = set()
        for dataset in logits_datasets:
            if dataset in seen:
                duplicates.add(dataset)
            else:
                seen.add(dataset)
        logger.warning(f"⚠️ Duplicate datasets found: {duplicates}")

    logits_list = []
    label_idx_list = []
    label_to_idx = {}

    labels = []

    for dataset_dir_i in logits_datasets:
        dataset_path_i = Path(dataset_dir_i)
        assert dataset_path_i.is_dir(), f"{dataset_path_i} is not a directory"
        with open(dataset_path_i / "label_to_idx.yaml", "r") as f:
            label_to_idx_i = yaml.safe_load(f)
        labels.extend(label_to_idx_i.keys())

    labels = sorted(list(set(labels)))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    logger.info(f"Merged label_to_idx:\n{pformat(label_to_idx, sort_dicts=False)}")

    for dataset_dir_i in logits_datasets:
        dataset_path_i = Path(dataset_dir_i)
        logits_i = np.load(dataset_path_i / "logits.npy")
        if truncate_first_n_tokens is not None:
            logits_i = logits_i[:, :truncate_first_n_tokens, :, :]
        if truncate_first_n_logits is not None:
            logits_i = logits_i[:, :, :truncate_first_n_logits, :]
        label_idx_i = np.load(dataset_path_i / "idx.npy")

        with open(dataset_path_i / "label_to_idx.yaml", "r") as f:
            label_to_old_idx_i = yaml.safe_load(f)

        old_idx_to_label_i = {i: label for label, i in label_to_old_idx_i.items()}

        @np.vectorize
        def update_idx(old_idx):
            return label_to_idx[old_idx_to_label_i[old_idx]]

        label_idx_i = update_idx(label_idx_i)

        logits_list.append(logits_i)
        label_idx_list.append(label_idx_i)
        logger.info(f"Loaded logits from {dataset_path_i}")

    if truncate_first_n_logits is None:
        truncate_first_n_logits = min([logits.shape[2] for logits in logits_list])
        logits_list = [logits[:, :, :truncate_first_n_logits, :] for logits in logits_list]

    logits = np.concatenate(logits_list, axis=0)
    label_idx = np.concatenate(label_idx_list, axis=0)

    if q_tags_to_remove is not None:
        q_tag_idx_to_remove = [label_to_idx[q_tag] for q_tag in q_tags_to_remove]
        mask = np.isin(label_idx, q_tag_idx_to_remove, invert=True)
        logits = logits[mask]
        label_idx = label_idx[mask]
        for q_tag in q_tags_to_remove:
            del label_to_idx[q_tag]

        new_label_to_idx = {label: idx for idx, label in enumerate(label_to_idx.keys())}

        @np.vectorize
        def update_idx(old_idx):
            return new_label_to_idx[idx_to_label[old_idx]]

        label_idx = update_idx(label_idx)
        label_to_idx = new_label_to_idx
        idx_to_label = {v: k for k, v in new_label_to_idx.items()}
        logger.info(f"Removed quantizer tags {q_tags_to_remove}")
        logger.info(f"Updated label_to_idx:\n{pformat(label_to_idx, sort_dicts=False)}")
    elif q_tags_to_keep is not None:
        q_tag_idx_to_keep = [label_to_idx[q_tag] for q_tag in q_tags_to_keep]
        mask = np.isin(label_idx, q_tag_idx_to_keep)
        logits = logits[mask]
        label_idx = label_idx[mask]
        q_tags_to_remove = [q_tag for q_tag in label_to_idx.keys() if q_tag not in q_tags_to_keep]
        for q_tag in q_tags_to_remove:
            del label_to_idx[q_tag]

        new_label_to_idx = {label: idx for idx, label in enumerate(label_to_idx.keys())}

        @np.vectorize
        def update_idx(old_idx):
            return new_label_to_idx[idx_to_label[old_idx]]

        label_idx = update_idx(label_idx)
        label_to_idx = new_label_to_idx
        idx_to_label = {v: k for k, v in new_label_to_idx.items()}
        logger.info(f"Only keeping quantizer tags {q_tags_to_keep}")
        logger.info(f"Updated label_to_idx:\n{pformat(label_to_idx, sort_dicts=False)}")

    unique, counts = np.unique(label_idx, return_counts=True)
    ratio = counts / counts.sum()
    labels_to_ratio = {idx_to_label[k]: round(v, 4) for k, v in zip(unique, ratio)}
    labels_to_count = {idx_to_label[k]: v for k, v in zip(unique, counts)}
    dataset_profile = {}
    for label in labels_to_ratio:
        dataset_profile[label] = f"{labels_to_count[label]} samples ({labels_to_ratio[label]:.2%})"
    logger.info(
        f"Logits dataset loaded ({label_idx.shape[0]} samples in total), label histogram:\n{pformat(dataset_profile, sort_dicts=False)}"
    )

    logits = np.ascontiguousarray(logits)
    label_idx = np.ascontiguousarray(label_idx)
    return logits, label_idx, label_to_idx, idx_to_label


"""
SVM
INFO    [llm_logits.py:689 | 2024-10-21_22:39:04] Training time: 9.0h 39.0m 17.34s
INFO    [llm_logits.py:700 | 2024-10-21_22:39:04] 🚀 Evaluating SVM on training set
INFO    [llm_logits.py:703 | 2024-10-21_22:39:04] Training accuracy: 0.4352
INFO    [llm_logits.py:708 | 2024-10-21_22:39:04] By-class accuracy:
{'NVIDIA-A100-SXM4-80GB_bf16': 0.69970703125,
 'NVIDIA-A100-SXM4-80GB_bf16-flashattn2': 0.38330078125,
 'NVIDIA-A100-SXM4-80GB_bf16-sdpa': 0.373779296875,
 'NVIDIA-A100-SXM4-80GB_fp16': 0.723388671875,
 'NVIDIA-A100-SXM4-80GB_fp16-flashattn2': 0.216796875,
 'NVIDIA-A100-SXM4-80GB_fp16-sdpa': 0.214111328125}
INFO    [llm_logits.py:711 | 2024-10-21_22:39:04] Training Matthews correlation: 0.3244
INFO    [llm_logits.py:720 | 2024-10-21_22:39:04] Training classification report:
{'NVIDIA-A100-SXM4-80GB_bf16': {'precision': 0.5008738203425376,
                                'recall': 0.69970703125,
                                'f1-score': 0.5838256264004889,
                                'support': 4096.0},
 'NVIDIA-A100-SXM4-80GB_bf16-flashattn2': {'precision': 0.3620013834447775,
                                           'recall': 0.38330078125,
                                           'f1-score': 0.3723467330724535,
                                           'support': 4096.0},
 'NVIDIA-A100-SXM4-80GB_bf16-sdpa': {'precision': 0.3646974749880896,
                                     'recall': 0.373779296875,
                                     'f1-score': 0.3691825415963347,
                                     'support': 4096.0},
 'NVIDIA-A100-SXM4-80GB_fp16': {'precision': 0.6147302904564316,
                                'recall': 0.723388671875,
                                'f1-score': 0.664647824136384,
                                'support': 4096.0},
 'NVIDIA-A100-SXM4-80GB_fp16-flashattn2': {'precision': 0.31444759206798867,
                                           'recall': 0.216796875,
                                           'f1-score': 0.2566473988439306,
                                           'support': 4096.0},
 'NVIDIA-A100-SXM4-80GB_fp16-sdpa': {'precision': 0.3278504672897196,
                                     'recall': 0.214111328125,
                                     'f1-score': 0.2590459311770787,
                                     'support': 4096.0},
 'accuracy': 0.4351806640625,
 'macro avg': {'precision': 0.4141001714315908,
               'recall': 0.4351806640625,
               'f1-score': 0.417616009204445,
               'support': 24576.0},
 'weighted avg': {'precision': 0.4141001714315908,
                  'recall': 0.4351806640625,
                  'f1-score': 0.417616009204445,
                  'support': 24576.0}}
"""


def train_svm(
    datasets: list[str],
    truncate_first_n_tokens: int = None,
    truncate_first_n_logits: int = None,
    q_tags_to_remove: list[str] = None,
    q_tags_to_keep: list[str] = None,
    svm_kernel: str = "linear",
    svm_n_iters: int = 1000,
    output_dir: str = None,
    seed: int = 1,
    **kwargs,
):
    if len(kwargs) > 0:
        raise ValueError(f"Unsupported arguments: {kwargs.keys()}")
    args = locals()
    logger.info(f"Arguments:\n{pformat(args)}")
    transformers.set_seed(seed)

    logger.info("🚀 Loading datasets")
    logits, label_idx, label_to_idx, idx_to_label = load_logit_datasets(
        logits_datasets=datasets,
        truncate_first_n_tokens=truncate_first_n_tokens,
        truncate_first_n_logits=truncate_first_n_logits,
        q_tags_to_remove=q_tags_to_remove,
        q_tags_to_keep=q_tags_to_keep,
    )

    logger.info("🚀 Training SVM")
    logits = logits.reshape(logits.shape[0], -1)  # [n_samples, n_tokens * first_n_logits_per_token * 3]
    # n_samples, n_tokens, n_logits, _ = logits.shape
    # logits = logits.reshape(n_samples * n_tokens, n_logits * 3)
    # label_idx = np.expand_dims(label_idx, axis=1).repeat(n_tokens, axis=1).flatten()
    # assert logits.shape[0] == label_idx.shape[0]

    scalar = StandardScaler()
    X = scalar.fit_transform(logits)
    X_train = X
    y_train = label_idx

    assert len(np.unique(label_idx)) > 1, "Only one class found in the dataset"

    start = time.time()
    if svm_kernel == "linear":
        svc = LinearSVC(random_state=seed, max_iter=svm_n_iters).fit(X_train, y_train)
    elif svm_kernel == "rbf":
        svc = SVC(kernel="rbf", max_iter=svm_n_iters, random_state=seed).fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported SVM kernel {svm_kernel}")
    logger.info(f"Training time: {smart_time(time.time() - start)}")

    if output_dir is not None:
        model_dir = Path(output_dir) / "svm_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "scalar.pkl", "wb") as f:
            pickle.dump(scalar, f)
        with open(model_dir / "svc.pkl", "wb") as f:
            pickle.dump(svc, f)
        logger.info(f"📦 SVM model saved to {model_dir}")

    logger.info("🚀 Evaluating SVM on training set")
    y_train_pred = svc.predict(X_train)
    acc_train = accuracy_score(y_train, y_train_pred)
    logger.info(f"Training accuracy: {acc_train:.4f}")

    confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
    class_acc = confusion_matrix_train.diagonal() / confusion_matrix_train.sum(axis=1)
    class_acc = {idx_to_label[i]: acc for i, acc in enumerate(class_acc.tolist())}
    logger.info(f"By-class accuracy:\n{pformat(class_acc, sort_dicts=False)}")

    matthews_corr_train = matthews_corrcoef(y_train, y_train_pred)
    logger.info(f"Training Matthews correlation: {matthews_corr_train:.4f}")

    classification_report_train = classification_report(
        y_train,
        y_train_pred,
        labels=np.unique(y_train),
        target_names=[idx_to_label[i] for i in np.unique(y_train)],
        output_dict=True,
    )
    logger.info(f"Training classification report:\n{pformat(classification_report_train, sort_dicts=False)}")

    if output_dir is not None:
        training_results = {
            "accuracy": acc_train,
            "classification_report": classification_report_train,
            "confusion_matrix": confusion_matrix_train.tolist(),
            "class_accuracy": class_acc,
            "matthews_correlation": matthews_corr_train.item(),
        }
        with open(model_dir / "training_results.yaml", "w") as f:
            yaml.dump(training_results, f)

        # env info
        env_info = get_pretty_env_info()
        env_info += f"\ntransformers version: {transformers.__version__}"
        with open(Path(model_dir).joinpath("env_info.txt"), "w") as f:
            f.write(env_info)

        # args
        with open(model_dir / "args.yaml", "w") as f:
            yaml.dump(args, f)

        logger.info(f"📦 Training results saved to {model_dir}")


def test_svm(
    datasets: list[str],
    model_dir: str,
    truncate_first_n_tokens: int = None,
    truncate_first_n_logits: int = None,
    q_tags_to_remove: list[str] = None,
    q_tags_to_keep: list[str] = None,
    output_dir: str = None,
    seed: int = 1,
):
    args = locals()
    logger.info(f"Arguments:\n{pformat(args)}")
    transformers.set_seed(seed)

    logits, label_idx, label_to_idx, idx_to_label = load_logit_datasets(
        logits_datasets=datasets,
        truncate_first_n_tokens=truncate_first_n_tokens,
        truncate_first_n_logits=truncate_first_n_logits,
        q_tags_to_remove=q_tags_to_remove,
        q_tags_to_keep=q_tags_to_keep,
    )

    logits = logits.reshape(logits.shape[0], -1)  # [n_samples, n_tokens * first_n_logits_per_token * 3]
    # n_samples, n_tokens, n_logits, _ = logits.shape
    # logits = logits.reshape(n_samples * n_tokens, n_logits * 3)
    # label_idx = np.expand_dims(label_idx, axis=1).repeat(n_tokens, axis=1).flatten()
    # assert logits.shape[0] == label_idx.shape[0]

    scalar_path = Path(model_dir) / "scalar.pkl"
    assert scalar_path.is_file(), f"{scalar_path} not found"
    with open(scalar_path, "rb") as f:
        scalar = pickle.load(f)

    svc_path = Path(model_dir) / "svc.pkl"
    assert svc_path.is_file(), f"{svc_path} not found"
    with open(svc_path, "rb") as f:
        svc = pickle.load(f)

    X = scalar.transform(logits)
    y_pred = svc.predict(X)

    # Test accuracy and other metrics

    y_true = label_idx
    acc = accuracy_score(y_true, y_pred)
    logger.info(f"Test accuracy: {acc:.4f}")

    confusion_matrix_test = confusion_matrix(y_true, y_pred)
    class_acc = confusion_matrix_test.diagonal() / confusion_matrix_test.sum(axis=1)
    class_acc = {idx_to_label[i]: acc for i, acc in enumerate(class_acc.tolist())}
    logger.info(f"By-class accuracy:\n{pformat(class_acc, sort_dicts=False)}")

    matthews_corr = matthews_corrcoef(y_true, y_pred)
    matthews_corr = matthews_corr.item()
    logger.info(f"Test Matthews correlation: {matthews_corr:.4f}")

    classification_report_test = classification_report(
        y_true,
        y_pred,
        labels=np.unique(y_true),
        target_names=[idx_to_label[i] for i in np.unique(y_true)],
        output_dict=True,
    )
    logger.info(f"Test classification report:\n{pformat(classification_report_test, sort_dicts=False)}")

    if output_dir is not None:
        output_dir = Path(output_dir).joinpath("test-svm")
        output_dir.mkdir(parents=True, exist_ok=True)
        test_results = {
            "accuracy": acc,
            "classification_report": classification_report_test,
            "confusion_matrix": confusion_matrix_test.tolist(),
            "class_accuracy": class_acc,
            "matthews_correlation": matthews_corr,
        }
        test_results_path = Path(output_dir) / "test_results.yaml"
        with open(test_results_path, "w") as f:
            yaml.dump(test_results, f)
        logger.info(f"📦 Test results saved to {test_results_path}")

        # env info
        env_info = get_pretty_env_info()
        env_info += f"\ntransformers version: {transformers.__version__}"
        with open(Path(output_dir) / "env_info.txt", "w") as f:
            f.write(env_info)

        # args
        with open(Path(output_dir) / "args.yaml", "w") as f:
            yaml.dump(args, f)


def visualize_bit_distribution(
    datasets: list[str],
    truncate_first_n_tokens: int = 2,
    truncate_first_n_logits: int = 4,
    q_tags_to_remove: list[str] = None,
    q_tags_to_keep: list[str] = None,
    output_dir: str = None,
    fig_size: tuple[int, int] = (5.5, 2.75),
):
    def process_q_name(name: str):
        if "fp16" in name:
            name = name.replace("fp16", "FP16")
        elif "bf16" in name:
            name = name.replace("bf16", "BF16")
        elif "fp32" in name:
            name = name.replace("fp32", "FP32")
        else:
            pass

        if "flashattn2" in name:
            name = name.replace("flashattn2", "FlashAttn2")
        elif "sdpa" in name:
            name = name.replace("sdpa", "SDPA")
        else:
            pass
        return name

    FONT_SIZE_ANNO = 8
    FONT_SIZE_S = 6
    FONT_SIZE_M = 10
    FONT_SIZE_L = 12

    plt.rc("font", size=FONT_SIZE_S)  # controls default text sizes
    plt.rc("axes", titlesize=FONT_SIZE_M)  # fontsize of the axes title
    plt.rc("axes", labelsize=FONT_SIZE_M)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=FONT_SIZE_S)  # fontsize of the tick labels
    plt.rc("legend", fontsize=FONT_SIZE_M)  # legend fontsize
    plt.rc("figure", titlesize=FONT_SIZE_L)  # fontsize of the figure title

    def concat_and_binarize(logits: np.ndarray):
        n_samples, n_tokens, n_logits, _ = logits.shape
        logits = logits.reshape(n_samples * n_tokens * n_logits, 3)
        sign = logits[:, 0].astype(np.int32) & 1
        exp = logits[:, 1].astype(np.int32) & 0xFF
        mantissa = logits[:, 2].astype(np.int32) & 0x7FFFFF
        fp32 = (sign << 31) | (exp << 23) | mantissa
        fp32 = fp32[:, np.newaxis]
        # unpack to bits
        view = fp32.view(np.uint8)
        if fp32.dtype.byteorder == ">" or (fp32.dtype.byteorder == "=" and sys.byteorder == "big"):
            view = view[::-1]

        bin_array = np.unpackbits(view, axis=1, bitorder="little")[:, ::-1]  # [n_samples * n_tokens * n_logits, 32]
        bin_array = bin_array.reshape(n_samples, n_tokens, n_logits, 32)
        return bin_array

    logits, label_idx, label_to_idx, idx_to_label = load_logit_datasets(
        logits_datasets=datasets,
        truncate_first_n_tokens=truncate_first_n_tokens,
        truncate_first_n_logits=truncate_first_n_logits,
        q_tags_to_remove=q_tags_to_remove,
        q_tags_to_keep=q_tags_to_keep,
    )
    # logits.shape = [n_samples, n_tokens, first_n_logits_per_token, 3]

    # split logits by label
    bin_list = []
    for idx in idx_to_label:
        mask = label_idx == idx
        lgt = logits[mask]
        bin_list.append(concat_and_binarize(lgt))

    # visualize
    if output_dir is not None:
        fig_dir = Path(output_dir) / "bit_distribution"
        fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=fig_size)
    bit_count_dict = {}
    n_samples, n_tokens, n_logits, _ = bin_list[0].shape
    for idx in idx_to_label:
        bin_array = bin_list[idx]
        bin_array = bin_array.reshape(n_samples, n_tokens * n_logits * 32).astype(np.int32)
        counts = bin_array.sum(axis=0)
        bit_count_dict[idx] = counts

    # bar plot
    count_min = np.min(np.stack(list(bit_count_dict.values()), axis=0), axis=0)
    bits_idx = np.arange(count_min.size) + 1
    for idx in bit_count_dict:
        counts = bit_count_dict[idx] - count_min
        ax.bar(bits_idx, counts, alpha=0.6, label=process_q_name(idx_to_label[idx]))

    x_ticks = np.arange(count_min.size + 1)
    ax.set_xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)
    ax.set_xticks(x_ticks[::32].tolist() + [count_min.size], minor=False)
    ax.set_xticks(x_ticks, minor=True)

    # annotate token id and logit id on x-axis
    anno_x = np.arange(16, count_min.size, 32).tolist()
    anno_text = [
        "$\mathrm[t_[{ti}]l_[{li}]]$".format(ti=ti, li=li).replace("[", "{").replace("]", "}")
        for ti in range(n_tokens)
        for li in range(n_logits)
    ]
    for t, x in zip(anno_text, anno_x):
        # text size is set to TEXT_SIZE_ANNO
        ax.annotate(
            t, (x, 0), xytext=(0, -6), textcoords="offset points", ha="center", va="top", fontsize=FONT_SIZE_ANNO
        )

    ax.legend(loc="upper right")
    ax.set_xlabel("Bit position")
    ax.set_ylabel("Bit count")
    # ax.set_title(f"Bit distribution")
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(fig_dir / f"bit_distribution.svg")
        plt.savefig(fig_dir / f"bit_distribution.pdf")
        logger.info(f"📦 Bit distribution saved to {fig_dir / 'bit_distribution.svg'}")


"""
hidden_size = 30000, 1 hidden layer
INFO    [llm_logits.py:1062 | 2024-10-19_16:09:52] Best Matthews correlation: 0.2254
INFO    [llm_logits.py:1063 | 2024-10-19_16:09:52] Best accuracy: 0.3430
INFO    [llm_logits.py:1064 | 2024-10-19_16:09:52] Best by-class accuracy:
{'NVIDIA-A100-SXM4-80GB_bf16': 0.848388671875,
 'NVIDIA-A100-SXM4-80GB_bf16-flashattn2': 0.059814453125,
 'NVIDIA-A100-SXM4-80GB_bf16-sdpa': 0.213623046875,
 'NVIDIA-A100-SXM4-80GB_fp16': 0.653564453125,
 'NVIDIA-A100-SXM4-80GB_fp16-flashattn2': 0.211181640625,
 'NVIDIA-A100-SXM4-80GB_fp16-sdpa': 0.071533203125}
"""


def train_mlp(
    datasets: list[str],
    truncate_first_n_tokens: int = None,
    truncate_first_n_logits: int = None,
    q_tags_to_remove: list[str] = None,
    q_tags_to_keep: list[str] = None,
    seed: int = 1,
    train_batch_size: int = 32,
    train_n_workers: int = 4,
    train_n_epochs: int = 100,
    train_lr: float = 1e-3,
    mlp_hidden_dim: int = 3000,
):
    transformers.set_seed(seed)

    class LogitsDatasetForMLP(Dataset):
        def __init__(self, logits: np.ndarray, label_idx: np.ndarray):
            self.logits = logits
            self.label_idx = label_idx

        def __len__(self):
            return self.logits.shape[0]

        def __getitem__(self, idx):
            return (
                torch.from_numpy(self.logits[idx]).float(),
                torch.tensor(self.label_idx[idx], dtype=torch.long),
            )

    logger.info("🚀 Loading datasets")
    logits, label_idx, label_to_idx, idx_to_label = load_logit_datasets(
        logits_datasets=datasets,
        truncate_first_n_tokens=truncate_first_n_tokens,
        truncate_first_n_logits=truncate_first_n_logits,
        q_tags_to_remove=q_tags_to_remove,
        q_tags_to_keep=q_tags_to_keep,
    )
    n_samples, n_tokens, n_logits, _ = logits.shape
    logits = logits.reshape(n_samples * n_tokens, n_logits * 3)
    label_idx = np.expand_dims(label_idx, axis=1).repeat(n_tokens, axis=1).flatten()
    assert logits.shape[0] == label_idx.shape[0]

    logits_dataset_train = LogitsDatasetForMLP(logits, label_idx)
    train_loader = DataLoader(
        logits_dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=train_n_workers
    )

    logger.info("🚀 Training SVM")
    logits = logits.reshape(logits.shape[0], -1)

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(MLP, self).__init__()
            self.bn = nn.BatchNorm1d(input_dim)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.bn(x)
            x = self.fc1(x)
            x = self.act1(x)
            x = self.fc2(x)
            return x

    mlp = MLP(input_dim=n_logits * 3, hidden_dim=mlp_hidden_dim, output_dim=len(label_to_idx))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=train_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    mlp = mlp.cuda()
    criterion = criterion.cuda()
    best_state_dict = None
    best_metric = None
    best_mett = -1
    y_true_list = None
    for epoch in tqdm.trange(train_n_epochs, desc="Training", unit="epoch"):
        mlp.train()
        y_true_list = []
        y_pred_list = []
        loss_list = []
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            y_pred = mlp(x)
            loss = criterion(y_pred, y)
            loss.backward()
            loss_list.append(loss.cpu().item())
            optimizer.step()
            y_true_list = y_true_list + y.cpu().tolist()
            y_pred_list = y_pred_list + y_pred.argmax(dim=1).cpu().tolist()

        lr_scheduler.step()
        acc = accuracy_score(y_true_list, y_pred_list)
        mett = matthews_corrcoef(y_true_list, y_pred_list)
        cm = confusion_matrix(y_true_list, y_pred_list)
        # by-class accuracy
        class_acc = cm.diagonal() / cm.sum(axis=1)
        class_acc = {idx_to_label[i]: acc for i, acc in enumerate(class_acc.tolist())}
        cls_report = classification_report(
            y_true_list,
            y_pred_list,
            labels=np.unique(y_true_list),
            target_names=[idx_to_label[i] for i in np.unique(y_true_list)],
            output_dict=True,
        )

        # report loss, acc, matthews corr
        avg_loss = np.mean(loss_list)
        logger.info(f"Epoch {epoch + 1}/{train_n_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Matthews: {mett:.4f}")

        if mett > best_mett:
            best_mett = mett
            best_state_dict = deepcopy(mlp.state_dict())
            best_metric = {
                "accuracy": acc,
                "classification_report": cls_report,
                "confusion_matrix": cm.tolist(),
                "class_accuracy": class_acc,
                "matthews_correlation": mett,
            }

    logger.info(f"Best Matthews correlation: {best_mett:.4f}")
    logger.info(f"Best accuracy: {best_metric['accuracy']:.4f}")
    logger.info(f"Best by-class accuracy:\n{pformat(best_metric['class_accuracy'], sort_dicts=False)}")
    logger.info(f"Best classification report:\n{pformat(best_metric['classification_report'], sort_dicts=False)}")


if __name__ == "__main__":
    set_logging_verbosity(logging.INFO)
    cli_map = {
        "collect": wrapped_create_logits_dataset,
        "train-svm": train_svm,
        "test-svm": test_svm,
        "vis-logits": visualize_bit_distribution,
        "train-mlp": train_mlp,
    }

    fire.Fire(cli_map)
