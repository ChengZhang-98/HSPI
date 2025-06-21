from argparse import ArgumentParser
from dataclasses import dataclass
import math
import random
from copy import deepcopy
import logging
import time
from pprint import pformat
from pathlib import Path
import yaml
import pickle
import os

import tqdm
import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.collect_env import get_pretty_env_info
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    HqqConfig,
)
from optimum.quanto import QuantizedModelForCausalLM, qint8
from transformers.utils.logging import set_verbosity_error as set_hf_verbosity_error
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from blackbox_locking.quantize import quantize_model


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("blackbox_locking." + __name__)


@dataclass
class RandomChatDatasetConfig:
    # args for generating random prompts
    num_samples: int
    num_random_tokens: int
    batch_size: int
    tokenizer_for_random_prompts: AutoTokenizer
    prompt: tuple[dict[str, str]]
    vocab_limit: tuple[int] = (None, None)


class RandomChatDataset:
    """
    HuggingFace ChatTemplate: https://huggingface.co/docs/transformers/v4.44.2/chat_templating#what-are-generation-prompts
    """

    def __init__(
        self,
        config: RandomChatDatasetConfig,
    ) -> None:
        self.config = config
        self.chats = []
        self.tokenized_chat_batches = None
        self._init_chats()

    def _init_chats(self):
        vocab_size = self.config.tokenizer_for_random_prompts.vocab_size
        vocab_lower_, vocab_upper_ = 0, vocab_size - 1
        vocab_lower, vocab_upper = self.config.vocab_limit
        if vocab_lower is not None:
            assert vocab_lower >= 0 and vocab_lower < vocab_size
            vocab_lower_ = vocab_lower
        if vocab_upper is not None:
            assert vocab_upper > 0 and vocab_upper < vocab_size
            vocab_upper_ = vocab_upper

        random_chats = []
        random_words = []
        for _ in range(self.config.num_samples):
            random_words.append(
                " ".join(
                    [
                        self.config.tokenizer_for_random_prompts.decode(random.randint(vocab_lower_, vocab_upper_))
                        for _ in range(self.config.num_random_tokens)
                    ]
                )
            )

        for i in range(self.config.num_samples):
            chat_i = deepcopy(self.config.prompt)
            chat_i[-1]["content"] += random_words[i]
            random_chats.append(chat_i)
        self.chats = random_chats

    def tokenize_chats(self, tokenizer):
        self.tokenized_chat_batches = []
        # decide seq length multiple of 8 that is greater than the longest chat
        tokenized_chat_0 = tokenizer.apply_chat_template(
            self.chats[0],
            tokenize=True,
            add_generation_prompt=False,  # we don't need this for now
            return_dict=True,
        )
        tokenized_chat_0_len = len(tokenized_chat_0["input_ids"])
        prompt_len = math.ceil(tokenized_chat_0_len / 8) * 8

        for i in range(self.config.num_samples // self.config.batch_size):
            tokenized_chats = tokenizer.apply_chat_template(
                self.chats[i * self.config.batch_size : (i + 1) * self.config.batch_size],
                tokenize=True,
                add_generation_prompt=False,  # we don't need this for now
                return_tensors="pt",
                return_dict=True,
                truncation=True,
                padding=True,
                max_length=prompt_len,
            )
            self.tokenized_chat_batches.append(tokenized_chats)

    def get_batch(self, i):
        assert self.tokenized_chat_batches is not None, "tokenize_chats() must be called first"
        assert i < len(self.tokenized_chat_batches), "batch index out of range"
        return self.tokenized_chat_batches[i]


@dataclass
class LogitsDatasetConfig:
    first_n_logits: int
    log_prob: bool

    start_token: int
    max_new_tokens: int

    batch_size: int = 10


default_generation_cfg = GenerationConfig(do_sample=False, num_beams=1, return_dict_in_generate=True)


def ndarray_to_bits(arr: np.ndarray):
    """
    convert a numpy array of float32 to a numpy array of int32 numbers representing the bits

    each fp32 number (byte3_byte2_byte1_byte0) is casted to 4 bytes: byte0, byte1, byte2, byte3
    each byte is then converted to 8 int32 numbers (0 or 1) representing the bits
    """
    assert arr.dtype == np.float32
    return np.unpackbits(arr.view(np.uint8), bitorder="little").astype(np.int32)


def build_model(model_name: str, q_tag: str):
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
    awq_ckpt_map = {
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    }
    if q_tag in ["awq", "bnb-8bit", "bnb-4bit", "hqq-4bit", "quanto-wi8ai8"]:
        if q_tag == "awq":
            assert model_name in awq_ckpt_map, f"AWQ checkpoint not found for model {model_name}"
            awq_ckpt = awq_ckpt_map[model_name]
            model = AutoModelForCausalLM.from_pretrained(awq_ckpt, trust_remote_code=True)
        elif q_tag in ["bnb-8bit", "bnb-4bit"]:
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
                model_name, torch_dtype=torch.float16, device_map="cuda", quantization_config=hqq_config
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
    elif q_tag in ["mxint8", "int8-dynamic", "fp8-e4m3", "fp8-e3m4"]:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
        model.eval()
        quantize_model(model, q_tag)
        model.cuda()
    else:
        raise ValueError(f"Unsupported quantizer tag {q_tag}")
    return model


@torch.no_grad()
def create_logits_dataset(
    model_names: list[str],
    q_tags: list[str],
    generation_cfg: GenerationConfig,
    random_chat_cfg: RandomChatDatasetConfig,
    logits_cfg: LogitsDatasetConfig,
):
    """
    num_samples = len(model_names) * len(q_tags) * random_chat_cfg.num_samples // logits_cfg.batch_size

    Returns:
    - logits (np.ndarray): shape [num_samples, logits_cfg.batch_size * logits_cfg.max_new_tokens * logits_cfg.first_n_logits]
    - idx_list (np.ndarray): shape [num_samples]
    - label_to_idx (dict[str, int]): a mapping from q_tag to its index in q_tags
    """
    assert random_chat_cfg.num_samples % logits_cfg.batch_size == 0, "num_samples must be divisible by batch_size"
    assert logits_cfg.start_token + logits_cfg.max_new_tokens <= generation_cfg.max_new_tokens
    # 2 + 2 = 4
    assert random_chat_cfg.batch_size == logits_cfg.batch_size
    num_batches = random_chat_cfg.num_samples // logits_cfg.batch_size
    logits: list[np.ndarray] = []
    idx_list: list[str] = []
    label_to_idx = {label: i for i, label in enumerate(q_tags)}

    chat_dataset = RandomChatDataset(random_chat_cfg)

    prog_bar = tqdm.tqdm(total=num_batches * len(model_names) * len(q_tags), desc="Generating logits dataset")

    model = None
    for model_name in model_names:
        for q_tag in q_tags:
            del model
            torch.cuda.empty_cache()
            model = build_model(model_name, q_tag)
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            chat_dataset.tokenize_chats(tokenizer)

            for i in range(num_batches):
                batch = chat_dataset.get_batch(i)
                inputs = batch.to("cuda")
                outputs = model.generate(**inputs, generation_config=generation_cfg)
                lgt = outputs.logits  # a tuple of [logits_cfg.batch_size, vocab_size]
                lgt = torch.stack(lgt, dim=1)  # [logits_cfg.batch_size, seq_len, vocab_size]
                assert lgt.shape[1] == generation_cfg.max_new_tokens
                assert lgt.shape[1] >= logits_cfg.start_token + logits_cfg.max_new_tokens
                lgt = lgt[
                    :,
                    logits_cfg.start_token : logits_cfg.start_token + logits_cfg.max_new_tokens,
                    : logits_cfg.first_n_logits,
                ]  # [logits_cfg.batch_size, logits_cfg.max_new_tokens, logits_cfg.first_n_logits]
                log_prob = torch.nn.functional.log_softmax(lgt, dim=-1)
                # shape: [logits_cfg.batch_size * logits_cfg.max_new_tokens * logits_cfg.first_n_logits]
                log_prob = log_prob.flatten().cpu().numpy().astype(np.float32)
                logits.append(ndarray_to_bits(log_prob))
                idx_list.append(label_to_idx[q_tag])
                prog_bar.update(1)

    logits = np.stack(logits)
    idx_list = np.array(idx_list)

    return logits, idx_list, label_to_idx, chat_dataset.chats


if __name__ == "__main__":
    set_hf_verbosity_error()
    DEFAULT_PROMPT = (
        {"role": "system", "content": "You are an unstable AI assistant."},
        {"role": "user", "content": "Please generate random words."},
        {"role": "system", "content": "Here are some random words: "},
    )
    parser = ArgumentParser()
    parser.add_argument("--model-names", type=str, nargs="+", default=["meta-llama/Meta-Llama-3.1-8B-Instruct"])
    parser.add_argument(
        "--q-tags",
        type=str,
        nargs="+",
        default=["bf16", "fp16"],
        choices=[
            "bf16",
            "fp16",
            "fp32",
            "awq",
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
            # these are emulated quantizers
            "mxint8",
            "int8-dynamic",
            "fp8-e4m3",
            "fp8-e3m4",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000, help="max iteration for training SVM")
    parser.add_argument("--request-batch-size", type=int, default=1, help="batch size for chat requests")
    parser.add_argument(
        "--num-raw-samples", type=int, default=2048, help="number of requests (un-batched samples)to generate"
    )
    parser.add_argument("--first-n-logits", type=int, default=128, help="number of logits to store for each request")
    parser.add_argument("--start-token", type=int, default=512, help="start token index for storing logits")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="max number of new tokens to generate")
    parser.add_argument(
        "--num-random-tokens", type=int, default=2, help="number of new random tokens appended to prompt"
    )
    parser.add_argument("--dataset-dir", type=str, nargs="+", default=None, help="load logits dataset from directories")
    parser.add_argument("--classifier-dir", type=str, default=None, help="load SVM classifier from directory")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="save logits dataset and SVM classifier to directory"
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="ratio of test set size")
    parser.add_argument(
        "--vocab-limit", type=int, nargs=2, default=(None, None), help="vocab lower and upper bound for random words"
    )
    parser.add_argument("--q-tags-to-remove", type=str, nargs="+", default=None, help="q_tags to remove from dataset")
    parser.add_argument(
        "--disable-device-prefix",
        action="store_true",
        help="disable device prefix in label_to_idx after logits generation",
    )
    # parser.add_argument("--num-workers", type=int, default=None, help="number of workers for SVM classifier")
    parser.add_argument("--svm-kernel", type=str, default="linear", help="SVM kernel", choices=["linear", "rbf"])
    args = parser.parse_args()

    logger.info(f"args: {pformat(vars(args))}")
    transformers.set_seed(args.seed)

    random_chat_cfg = RandomChatDatasetConfig(
        num_samples=args.num_raw_samples,
        num_random_tokens=args.num_random_tokens,
        batch_size=args.request_batch_size,
        tokenizer_for_random_prompts=AutoTokenizer.from_pretrained(
            args.model_names[0], padding_side="left", trust_remote_code=True
        ),
        vocab_limit=args.vocab_limit,
        prompt=DEFAULT_PROMPT,
    )

    logits_cfg = LogitsDatasetConfig(
        first_n_logits=args.first_n_logits,
        log_prob=True,
        start_token=args.start_token,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.request_batch_size,
    )
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=True,
        output_logits=True,
        max_new_tokens=args.start_token + args.max_new_tokens,
        min_new_tokens=args.start_token + args.max_new_tokens,
    )

    if args.dataset_dir is None:
        logger.info("🚀 Generating logits dataset...")
        start = time.time()
        logits, label_idx, label_to_idx, requests = create_logits_dataset(
            model_names=args.model_names,
            q_tags=args.q_tags,
            generation_cfg=generation_config,
            random_chat_cfg=random_chat_cfg,
            logits_cfg=logits_cfg,
        )
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        # add device as label prefix
        if not args.disable_device_prefix:
            device_prefix = torch.cuda.get_device_name().replace(" ", "-")
            label_to_idx = {f"{device_prefix}_{k}": v for k, v in label_to_idx.items()}
            idx_to_label = {v: k for k, v in label_to_idx.items()}
        logger.info(f"Logits dataset generation time: {time.time() - start:.2f}s")
    else:
        logger.info("🚀 Loading logits dataset...")
        logits_list = []
        label_idx_list = []
        label_to_idx = {}
        requests = []

        labels = []
        for dataset_dir_i in args.dataset_dir:
            dataset_path_i = Path(dataset_dir_i)
            assert dataset_path_i.is_dir(), f"{dataset_path_i} does not exist"
            with open(dataset_path_i.joinpath("label_to_idx.yaml"), "r") as f:
                label_to_old_idx_i = yaml.safe_load(f)
            labels.extend(label_to_old_idx_i.keys())

        labels = sorted(list(set(labels)))
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        logger.info(f"Merged label_to_idx:\n{pformat(label_to_idx, sort_dicts=False)}")

        for dataset_dir_i in args.dataset_dir:
            dataset_path_i = Path(dataset_dir_i)
            logits_i = np.load(dataset_path_i.joinpath("logits.npy"))
            label_idx_i = np.load(dataset_path_i.joinpath("idx.npy"))
            with open(dataset_path_i.joinpath("label_to_idx.yaml"), "r") as f:
                label_to_old_idx_i = yaml.safe_load(f)
            with open(dataset_path_i.joinpath("requests.yaml"), "r") as f:
                requests_i = yaml.safe_load(f)
            requests.extend(requests_i)

            old_idx_to_label_i = {v: k for k, v in label_to_old_idx_i.items()}

            def update_idx(old_idx):
                return label_to_idx[old_idx_to_label_i[old_idx]]

            update_idx = np.vectorize(update_idx)
            label_idx_i = update_idx(label_idx_i)

            logits_list.append(logits_i)
            label_idx_list.append(label_idx_i)
            logger.info(f"Logits dataset loaded from {dataset_path_i}")

        # check shape[1] of all logits
        assert all(logits.shape[1] == logits_list[0].shape[1] for logits in logits_list)

        logits = np.concatenate(logits_list, axis=0)
        label_idx = np.concatenate(label_idx_list, axis=0)

    if args.q_tags_to_remove is not None:
        q_tags_to_remove = args.q_tags_to_remove
        q_tags_to_remove_idx = [label_to_idx[q_tag] for q_tag in q_tags_to_remove]
        mask = np.isin(label_idx, q_tags_to_remove_idx, invert=True)
        logits = logits[mask]
        label_idx = label_idx[mask]
        for q_tag in q_tags_to_remove:
            del label_to_idx[q_tag]

        new_label_to_idx = {label: idx for idx, label in enumerate(label_to_idx.keys())}

        def update_idx(old_idx):
            return new_label_to_idx[idx_to_label[old_idx]]

        update_idx = np.vectorize(update_idx)
        label_idx = update_idx(label_idx)
        idx_to_label = {v: k for k, v in new_label_to_idx.items()}
        label_to_idx = new_label_to_idx
        logger.info(f"Removed q_tags: {q_tags_to_remove}")
        logger.info(f"Updated label_to_idx:\n{pformat(label_to_idx, sort_dicts=False)}")

    unique, counts = np.unique(label_idx, return_counts=True)
    ratio = counts / counts.sum()
    labels_to_ratio = {idx_to_label[k]: round(v, 4) for k, v in zip(unique, ratio)}
    labels_to_count = {idx_to_label[k]: v for k, v in zip(unique, counts)}
    dataset_profile = {}
    for label in labels_to_ratio:
        dataset_profile[label] = f"{labels_to_count[label]} samples ({labels_to_ratio[label]:.2%})"
    logger.info(
        f"Logits dataset loaded ({label_idx.shape[0]} samples in total), label distribution:\n{pformat(dataset_profile, sort_dicts=False)}"
    )

    if args.output_dir is not None:
        dataset_dir = Path(args.output_dir).joinpath("logits_dataset")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.save(dataset_dir.joinpath("logits.npy"), logits)
        np.save(dataset_dir.joinpath("idx.npy"), label_idx)
        with open(dataset_dir.joinpath("label_to_idx.yaml"), "w") as f:
            yaml.safe_dump(label_to_idx, f)
        with open(dataset_dir.joinpath("requests.yaml"), "w") as f:
            yaml.safe_dump(requests, f)

        logger.info(f"Logits dataset saved to {dataset_dir}")
    else:
        logger.warning("Logits dataset not saved, specify --output-dir to save")

    logits = np.ascontiguousarray(logits)
    label_idx = np.ascontiguousarray(label_idx)
    X_train, X_test, y_train, y_test = train_test_split(
        logits, label_idx, test_size=args.test_size, random_state=args.seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if len(np.unique(y_train)) == 1:
        logger.warning("⚠️ Only one class in training set. The classifier may not be able to learn.")

    svc = None
    if args.classifier_dir is None and len(np.unique(y_train)) > 1:
        logger.info("🚀 Training SVM classifier...")
        if args.svm_kernel == "rbf":
            svc = SVC(kernel="rbf", random_state=args.seed, max_iter=args.max_iter).fit(X_train, y_train)
        elif args.svm_kernel == "linear":
            svc = LinearSVC(random_state=args.seed, max_iter=args.max_iter).fit(X_train, y_train)
        else:
            raise ValueError(f"Unsupported SVM kernel {args.svm_kernel}")
        # save model
        if args.output_dir is not None:
            model_dir = Path(args.output_dir).joinpath("svm_model")
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir.joinpath("scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            with open(model_dir.joinpath("svc.pkl"), "wb") as f:
                pickle.dump(svc, f)
            logger.info(f"SVM model saved to {model_dir}")
    elif len(np.unique(y_train)) <= 1:
        logger.warning("⚠️ Only one class in training set. Skip training SVM classifier.")
    else:
        logger.info("🚀 Loading SVM classifier...")
        model_path = Path(args.classifier_dir)
        assert model_path.is_dir()
        with open(model_path.joinpath("scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(model_path.joinpath("svc.pkl"), "rb") as f:
            svc = pickle.load(f)
        logger.info(f"SVM model loaded from {model_path}")

    # evaluate
    classification_report_dict = None
    if svc is not None:
        logger.info("🚀 Evaluating SVM classifier...")
        # eval on training set

        y_train_pred = svc.predict(X_train)
        acc_train = accuracy_score(y_train, y_train_pred)
        logger.info(f"{type(svc).__name__}, training accuracy: {acc_train} on {y_train.shape[0]} samples")
        classification_report_str = classification_report(
            y_train, y_train_pred, labels=np.unique(y_train), target_names=[idx_to_label[i] for i in np.unique(y_train)]
        )
        classification_report_dict = classification_report(
            y_train,
            y_train_pred,
            labels=np.unique(y_train),
            target_names=[idx_to_label[i] for i in np.unique(y_train)],
            output_dict=True,
        )

        logger.info(f"Classification report:\n{classification_report_str}")
        # confusion matrix
        cm = confusion_matrix(y_train, y_train_pred)
        # by-class accuracy
        class_acc = cm.diagonal() / cm.sum(axis=1)
        logger.info(
            f"By-class accuracy:\n{pformat({idx_to_label[i]: acc for i, acc in enumerate(class_acc)}, sort_dicts=False)}"
        )

        # eval on test set
        # y_pred = svc.predict(X_test)
        # acc = accuracy_score(y_test, y_pred)
        # logger.info(f"{type(svc).__name__}, test accuracy: {acc} on {y_test.shape[0]} samples")
        # classification_report_str = classification_report(
        #     y_test, y_pred, labels=np.unique(y_test), target_names=[idx_to_label[i] for i in np.unique(y_test)]
        # )
        # classification_report_dict = classification_report(
        #     y_test,
        #     y_pred,
        #     labels=np.unique(y_test),
        #     target_names=[idx_to_label[i] for i in np.unique(y_test)],
        #     output_dict=True,
        # )
        # logger.info(f"Classification report:\n{classification_report_str}")
        # # confusion matrix
        # cm = confusion_matrix(y_test, y_pred)
        # # by-class accuracy
        # class_acc = cm.diagonal() / cm.sum(axis=1)
        # logger.info(
        #     f"By-class accuracy:\n{pformat({idx_to_label[i]: acc for i, acc in enumerate(class_acc)}, sort_dicts=False)}"
        # )

    # save args
    if args.output_dir is not None:
        with open(Path(args.output_dir).joinpath("args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), f)

        # env info
        env_info = get_pretty_env_info()
        env_info += f"\ntransformers version: {transformers.__version__}"
        with open(Path(args.output_dir).joinpath("env_info.txt"), "w") as f:
            f.write(env_info)

        # save evaluation results
        if classification_report_dict is not None:
            with open(Path(args.output_dir).joinpath("classification_report.yaml"), "w") as f:
                yaml.safe_dump(classification_report_dict, f)

            # save confusion matrix
            np.save(Path(args.output_dir).joinpath("confusion_matrix.npy"), cm)

        # save requests
        with open(Path(args.output_dir).joinpath("requests.yaml"), "w") as f:
            yaml.safe_dump(requests, f)

        # save profile
        with open(Path(args.output_dir).joinpath("dataset_profile.yaml"), "w") as f:
            yaml.safe_dump(dataset_profile, f)
        logger.info(f"Dataset, SVM, requests, and metrics saved to {args.output_dir}")
