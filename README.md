# README

## Overview

This code accompanies the paper [Hardware and Software Platform Inference](https://openreview.net/pdf?id=kdmjVF1iDO). Our method identifies the underlying GPU architecture and software stack of a (black-box) machine learning model solely based on its input–output behavior. We leverage the inherent differences of various GPU architectures and compilers to distinguish between different GPU types and software stacks. We evaluate HSPI against models served on real hardware and achieve between 83.9% and 100% accuracy in a white-box setting and up to 3× higher than random-guess accuracy in a black-box setting.

## Environment

Just run: conda env create -f environment.yml

- **Conda env:** `env.yaml`
- **Pip requirements:** `requirements.txt`

If you want to specifically differentiate and use different cuda versions, please set these in env.yaml. When doing border image generation across GPUs be careful to ensure that the compatible version of nccl are installed (best if the same version).

## Main Experiments

### HSPI-BI: Creating Border Images between quantization classes via PGD

#### Script: `img_border_batched_quantization.py`

Use this script to differentiate different quantization schemes on the same GPU by generating "border images" via projected gradient descent (PGD). It supports both single-image and batched operation modes, configurable learning-rate schedules, and optional transferability checks against additional models.

When transferability checks are enabled (using the `--other-models` flags), the script evaluates how each crafted border image generalizes to the specified additional models. It computes a transferability ratio per image and outputs detailed metadata plus a summary file (`transferability.yaml`) showing how often predicted quantization tags match the true tags across those models.

##### Usage

```bash
python img_border_batched_quantization.py `<model_name>` `<dataset>` `<save_dir>` [options]
```

- `<model_name>`: one of `vgg16`, `resnet18`, `resnet50`, `efficientnet_b0`, `densenet121`, `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`
- `<dataset>`: one of `cifar10`, `cifar100`, `imagenet`
- `<save_dir>`: directory where checkpoints, border images, and logs will be saved

##### Key Options

- `--method`: Engineering method; only `1-vs-1-pgd` is implemented (default: `1-vs-1-pgd`).

- **Fine-tuning**

  - `--fine-tune-batch-size`: batch size for initial FP32 fine-tuning (default: `128`).
  
- **PGD Engineering**

  - `--engineer-batch-size`: batch size for border-image crafting (use `1` to emulate single-image mode) (default: `32`).
  - `--start-lr`, `--end-lr`: linear learning-rate schedule start and end values (default: `1e-3`, `1e-4`).
  - `--num-iters`: number of PGD iterations per image (default: `400`).
  - `--check-every`: integer or ratio to check for border images during PGD (default: ratio `None` which skips intermediate checks).
  - `--noise-scale`: initial random noise scale added to source images (default: `0.01`).

- **Quantization Tags**

  - `--q-tags`: list of quantization schemes to evaluate (e.g. `fp32 bf16 fp16 mxint8 fp8-e3m4 fp8-e4m3 int8-dynamic`).

- **Transferability (optional)**

  - `--other-models`: list of other model names to test border-image transferability.
  - `--other-model-ckpts`: corresponding checkpoint paths for the other models.
  - `--other-model-tags`: tags for each of the other models (defaults to `--q-tags`).

- **Miscellaneous**

  - `--device`: computation device (e.g. `cuda`, `cpu`).
  - `--seed`: random seed (default: `42`).
  - `--overwrite` / `-ow`: overwrite existing `save_dir` if it exists.
  - `--skip-q-test`: skip quantized-model accuracy evaluation.
  - `--create-model-ckpt-only`: exit after saving the FP32 model checkpoint, without engineering border images.

##### Example

```bash
python img_border_batched_quantization.py resnet18 cifar10 results/BI_quant_resnet18 \
  --method 1-vs-1-pgd \
  --fine-tune-batch-size 128 \
  --engineer-batch-size 32 \
  --num-iters 200 --check-every 20 \
  --q-tags fp32 bf16 fp16 \
  --device cuda
```

If transferability check to another model is required:

```bash
python img_border_batched_quantization.py resnet18 cifar10 results/BI_quant_resnet18_transfer_resnet50 \
  --method 1-vs-1-pgd \
  --fine-tune-batch-size 128 \
  --engineer-batch-size 32 \
  --num-iters 200 --check-every 20 \
  --q-tags fp32 bf16 fp16 \
  --device cuda
  --other-models resnet50
```

### HSPI-BI: Creating Border Images between two GPUs via PGD

Before you run the main border-image script in distributed (NCCL) mode, you can verify your NCCL connectivity:

```bash
# On your “master” node:
python nccl_simple_test.py \
  --rank 0 \
  --world_size 2 \
  --master_addr 111.111.11.11 \
  --master_port 22222

# On the second node:
python nccl_simple_test.py \
  --rank 1 \
  --world_size 2 \
  --master_addr 111.111.11.11 \
  --master_port 22222
```

If both ranks report correct all\_reduce sums, your network & NCCL setup is good. If connection does not work, it might be nccl version comptability or firewall rules. Check firewall rules with sudo ufw status.

---

#### Launching `img_border_batched_GPUs.py`

Once NCCL is working, run the *border-image* script on **both** machines, pointing at the same master IP & port. You’ll also need to set `NCCL_SOCKET_IFNAME` if your machines have multiple NICs:

```bash
# On the “master” server (rank 0):
NCCL_SOCKET_IFNAME=eno1 \
python img_border_batched_GPUs.py \
  --rank 0 \
  --world_size 2 \
  --master_addr 111.111.11.11 \
  --master_port 22222 \
  --model_name resnet18 \
  --dataset cifar10 \
  --save_dir results/border_images_batchsize \
  [other options…]


# On the second server (rank 1):
NCCL_SOCKET_IFNAME=eth0 \
python img_border_batched_GPUs.py \
  --rank 1 \
  --world_size 2 \
  --master_addr 111.111.11.11 \
  --master_port 22222 \
  --model_name resnet18 \
  --dataset cifar10 \
  --save_dir results/border_images_batchsize \
  [same other options…]
```

Be sure that:

1. `--master_addr` & `--master_port` match on both calls
2. `rank` is `0` on the master, `1` on the second node
3. `world_size` is `2` on both nodes
4. You use the correct NIC name for `NCCL_SOCKET_IFNAME` on each machine

Other options are similar to the quantization version.

### HSPI-LD: Logit‐Distribution Fingerprinting

A fast, gradient‐free way to tell quantization schemes apart by training an SVM on the bit‐patterns of a model’s logits.

#### 1. `img_logits_svm.py`

Collects logits, converts them to 32-bit IEEE-754 bit patterns, groups them into blocks, then trains and evaluates a One-vs-One `LinearSVC` **within** a single model.

**Usage**

```bash
python img_logits_svm.py \
  --model-ckpt checkpoints/resnet50-cifar10.pt \
  --save-dir results/img_logits_svm \
  --q-tags fp32 bf16 fp16 int8-dynamic \
  --num-samples 5000 \
  --no-logits 10 \
  --eval-batch-size 128 \
  --device cuda
```

- **`--model-ckpt`**: Path to your model checkpoint (fine-tuned on CIFAR-10).
- **`--save-dir`**: Where to store (and/or load)
  `uniform_logits_… .pth` and `uniform_y_labels_… .pth`.
- **`--q-tags`**: Quantization schemes to compare.
- **`--num-samples`**: Number of images per quantization tag.
- **`--no-logits`**: How many logits make up one SVM example.
- **`--eval-batch-size`, `--device`, `--num-workers`**: Standard DataLoader & compute settings.

The script will:

1. Generate or load `uniform_logits_<…>.pth` & `uniform_y_labels_<…>.pth`.
2. Bit-encode and block-group the logits.
3. Split 80/20, train a One-vs-One `LinearSVC`.
4. Print within-model SVM accuracy & full classification report.

#### 2. `img_logits_svm_transferability.py`

Trains on one model’s logits, then **tests** on another model’s logits to measure cross-model robustness of the logit fingerprint.

**Usage**

```bash
python img_logits_svm_transferability.py \
  --model_ckpt1 checkpoints/mobilenet-v2-cifar10.pt \
  --model_ckpt2 checkpoints/efficientnet-b0-cifar10.pt \
  --save-dir results/img_logits_svm_transfer \
  --q-tags fp32 bf16 fp16 \
  --num-samples 5000 \
  --no-logits 10 \
  --eval-batch-size 64 \
  --device cuda
```

- **`--model_ckpt1` / `--model_ckpt2`**:
  Checkpoints for **train** and **test** models.
- **`--logits_file1` / `--logits_file2`** (optional):
  Pre-computed `uniform_logits_…` files to reuse.
- All other flags match `img_logits_svm.py`.

This script will:

1. Load or generate logits for both models.
2. Bit-encode & block-group each.
3. Train the SVM on model 1’s blocks.
4. Evaluate that classifier on model 2’s blocks.
5. Print **transfer** accuracy & classification report.

#### 3. `img_logits_svm_transferability_gpus.py`

Extends the transferability test to distinguish not only between quantization schemes and model architectures, but also between **different GPU hardware**. You supply two (optionally different if transferability needs to be checked) model checkpoints and train an SVM on the bit-patterns of one GPU’s logits, then test on the other GPU’s logits. The logits from the two different GPU's can be prepared in advance by running logits_svm.py on each GPU and then sharing the produced logits and running this script pointing at these files. If the exact same model should be used to compute the logits on both GPUs, transfer the model between servers in advance.

**Usage**

```bash
python img_logits_svm_transferability_gpus.py \
  --model_ckpt1 checkpoints/mobilenet-v2-cifar10.pt \
  --model_ckpt2 checkpoints/mobilenet-v2-cifar10.pt \
  --logits_file1 checkpoints/RTX8000uniform_logits_{args_num_samples}_{args.no_logits}_{model_ckpt_base}_{q_tags_str}.pth \
  --logits_file2 checkpoints/A100uniform_logits_{args_num_samples}_{args.no_logits}_{model_ckpt_base}_{q_tags_str}.pth \
  --save_dir results/img_logits_svm_transfer_gpus \
  --q-tags fp32 bf16 fp16 \
  --num-samples 5000 \
  --no-logits 10 \
  --eval-batch-size 128 \
  --device cuda:0
```

## SGL experiments

Scripts of HSPI-LD experiments for [SGL](https://docs.sglang.ai/index.html) can be found under [sgl-hspi-ld](/sgl-hspi-ld).
This requires launching sgl server first before collecting logits.

1. Please refer to SGL's docs to install SGL properly.

2. Launch SGL server first. You may use SGL CLI args like `--tp 2`, `--dp 2`, `--attention-backend flashinfer` etc to enable tensor parallel, data parallel, specify kernel backend, etc.

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct
  ```

3. Run the collect script to generate & save logits

  ```bash
  cd sgl-hspi-ld
  # check usage
  python collect.py collect --help
  # pass a config file instead of specify args
  python collect.py collect --config ./config --save_dir ./path/to/save/logits
  ```

4. Run the classifier to train an SVM to predict SW/HW stack.
  
  ```bash
  # check CLI actions and usage 
  python classify.py --help
  # train svm
  python classify.py train-svm ./path/to/saved-logits
  ```

### SGL HSPI-LD dataset

You can find example logits collected by `collect.py` at [Cheng98/HSPI-SGL](https://huggingface.co/datasets/Cheng98/HSPI-SGL)
