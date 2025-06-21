import logging
from copy import deepcopy
from pprint import pformat

import torch
from torch.utils.data import DataLoader
import transformers

from blackbox_locking.quantize import quantize_model
from blackbox_locking.datasets import get_wikitext2
from blackbox_locking.evaluate import evaluate_perplexity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad
def test_quantizers():
    """
    {
        "fp32": 0.0,
        "fp16": 8.632293701171875,
        "bf16": 0.10015921294689178,
        "bypass": 0.0,
        "int8-dynamic": 8.567744255065918,
        "fp8-e4m3": 8.467926979064941,
        "fp8-e3m4": 8.755851745605469,
        "mxint8": 0.3927737772464752,
        "bm8": 3.299532413482666,
        "bl8": 2.6694018840789795,
        "log8": 2.6694018840789795,
    }
    """
    model_name = "TinyLlama/TinyLlama_v1.1"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    input_text = "Hello, my dog is cute"
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].cuda()

    q_errors = {}

    model_ref = transformers.AutoModelForCausalLM.from_pretrained(model_name, _attn_implementation="eager")
    model_ref.eval()
    model_ref.to("cuda")
    logits_ref = model_ref(input_ids).logits
    model_ref.to("cpu")

    for q_name in [
        # built-in:
        "fp32",
        "fp16",
        "bf16",
        "bypass",
        # emulated:
        ## relatively precise
        "int8-dynamic",
        "fp8-e4m3",
        "fp8-e3m4",
        "mxint8",
        ## may cause significant error
        "bm8",
        "bl8",
        "log8",
    ]:
        model_q = deepcopy(model_ref)
        quantize_model(model_q, q_name)

        model_q.eval()
        model_q.to("cuda")

        logits_q = model_q(input_ids).logits
        q_errors[q_name] = (logits_ref - logits_q).abs().mean().cpu().item()

        del model_q

    logger.info(pformat(q_errors, sort_dicts=False))


@torch.no_grad
def test_quantizers_ppl():
    """
    {'reference': {'loss': 2.043262897468195,
                   'perplexity': 7.715743851760853,
                   'num_samples': 164,
                   'seq_len': 2048,
                   'batch_size': 4},
     'fp32': {'loss': 2.043262897468195,
              'perplexity': 7.715743851760853,
              'num_samples': 164,
              'seq_len': 2048,
              'batch_size': 4},
     'fp16': {'loss': 2.0433229673199538,
              'perplexity': 7.716207349271213,
              'num_samples': 164,
              'seq_len': 2048,
              'batch_size': 4},
     'bf16': {'loss': 2.043629024086929,
              'perplexity': 7.718569308174157,
              'num_samples': 164,
              'seq_len': 2048,
              'batch_size': 4},
     'bypass': {'loss': 2.043262897468195,
                'perplexity': 7.715743851760853,
                'num_samples': 164,
                'seq_len': 2048,
                'batch_size': 4},
     'int8-dynamic': {'loss': 3.687596239694735,
                      'perplexity': 39.948704341934764,
                      'num_samples': 164,
                      'seq_len': 2048,
                      'batch_size': 4},
     'fp8-e4m3': {'loss': 2.1160032807326896,
                  'perplexity': 8.297906721229351,
                  'num_samples': 164,
                  'seq_len': 2048,
                  'batch_size': 4},
     'fp8-e3m4': {'loss': 3.5418823870216927,
                  'perplexity': 34.531860375265396,
                  'num_samples': 164,
                  'seq_len': 2048,
                  'batch_size': 4},
     'mxint8': {'loss': 2.0446775570148374,
                'perplexity': 7.7266667267132325,
                'num_samples': 164,
                'seq_len': 2048,
                'batch_size': 4}}
    """
    model_name = "TinyLlama/TinyLlama_v1.1"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    data_module = get_wikitext2(tokenizer, max_length=2048)
    dataloader = DataLoader(
        data_module["test"],
        batch_size=4,
        shuffle=False,
        collate_fn=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        num_workers=8,
    )

    model_ref = transformers.AutoModelForCausalLM.from_pretrained(model_name, _attn_implementation="eager")
    model_ref.eval()
    model_ref.to("cuda")
    ppl_ref = evaluate_perplexity(model_ref, dataloader, progress_bar=True, description="reference")
    model_ref.to("cpu")
    ppl_dict = {"reference": ppl_ref}

    for q_name in [
        # built-in:
        "fp32",
        "fp16",
        "bf16",
        "bypass",
        # emulated:
        ## relatively precise
        "int8-dynamic",
        "fp8-e4m3",
        "fp8-e3m4",
        "mxint8",
        ## may cause significant error
        # "bm8",
        # "bl8",
        # "log8",
    ]:
        model_q = deepcopy(model_ref)
        quantize_model(model_q, q_name)

        model_q.eval()
        model_q.to("cuda")

        ppl_q = evaluate_perplexity(model_q, dataloader, progress_bar=True, description=q_name)
        ppl_dict[q_name] = ppl_q

        del model_q

    logger.info(pformat(ppl_dict, sort_dicts=False))


if __name__ == "__main__":
    # test_quantizers()
    test_quantizers_ppl()
