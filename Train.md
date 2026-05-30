# PixZig Field Training

This document describes how to train PixZig Field with the `pixzig_field.train`
CLI, how the dataset is expected to be organized, what files are produced, and
what every training argument controls.

## Dataset

The default dataset format is an image folder with one text caption file per
image. Caption files use the same base name as the image.

```text
dataset/
  image_000001.png
  image_000001.txt
  image_000002.jpg
  image_000002.txt
  subfolder/
    image_000003.webp
    image_000003.txt
```

Supported image extensions are handled by Pillow and include common formats
such as `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

By default, images without captions are skipped. Empty captions are skipped
unless `--allow-empty-text` is enabled. Recursive loading is enabled by default
and can be disabled with `--non-recursive-data`.

Caption metadata can also be provided through a `.jsonl` or `.csv` file with
`--captions`. Supported field names include image path fields such as
`image_path`, `file_name`, or `image`, and caption fields such as `text` or
`caption`.

Images are converted to RGB, resized/cropped to `--image-size`, normalized to
`[-1, 1]`, and returned with their text caption.

## Running Training

Basic command:

```bash
python -m pixzig_field.train \
  --data ./dataset \
  --output ./runs/pixzig-field \
  --device cuda \
  --image-size 512 \
  --patch-size 16 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --precision bf16 \
  --text-encoder qwen \
  --qwen-model-name ./Qwen3.5-2B-Base \
  --qwen-cache-embeddings \
  --qwen-max-length 128 \
  --text-encoder-device cuda \
  --cond-dim 1024 \
  --max-steps 19060 \
  --periodic-checkpoints \
  --save-every-steps 1000 \
  --save-safetensors \
  --periodic-samples \
  --sample-every-steps 1000 \
  --sample-n-steps 20 \
  --sample-method euler
```

Unconditional training:

```bash
python -m pixzig_field.train \
  --data ./dataset \
  --output ./runs/pixzig-unconditional \
  --device cuda \
  --text-encoder none \
  --allow-empty-text
```

## Step Count

`--max-steps` counts optimizer steps, not individual images.

```text
effective_batch_size = batch_size * gradient_accumulation_steps
steps_per_epoch = number_of_images / effective_batch_size
```

Example with 7,694 images:

```text
batch_size=1, gradient_accumulation_steps=1 -> 7,694 steps per epoch
batch_size=1, gradient_accumulation_steps=2 -> 3,847 steps per epoch
batch_size=2, gradient_accumulation_steps=1 -> 3,847 steps per epoch
batch_size=1, gradient_accumulation_steps=4 -> 1,924 steps per epoch
```

## Resuming

Use `--resume` with a `.pt` checkpoint, normally `last.pt`. A `.pt` checkpoint
contains model weights, EMA state, optimizer state, scheduler state, scaler
state, step, epoch, and configuration metadata.

```bash
python -m pixzig_field.train \
  --data ./dataset \
  --output ./runs/pixzig-field \
  --resume ./runs/pixzig-field/checkpoints/last.pt \
  --device cuda \
  --max-steps 38120
```

When resuming, `--max-steps` is the new total step limit, not the number of
extra steps to add.

## Outputs

The training command writes the run configuration, checkpoints, optional
`.safetensors` exports, and optional sample images under `--output`.

```text
runs/pixzig-field/
  run_config.json
  checkpoints/
    last.pt
    last.pt.json
    model.safetensors
    model.safetensors.json
    step_00001000.pt
    step_00001000.pt.json
    step_00001000.safetensors
    step_00001000.safetensors.json
  samples/
    step_00001000.png
    final.png
```

## Argument Reference

### Configuration

| Argument | Description |
| --- | --- |
| `--config PATH` | Optional JSON run config. Sections in this file can override CLI-derived configs. |

### Dataset

| Argument | Description |
| --- | --- |
| `--data PATH` | Required dataset root directory. |
| `--captions PATH` | Optional `.jsonl` or `.csv` captions file. |
| `--caption-extension EXT` | Caption sidecar extension for image-pair datasets. Default: `.txt`. |
| `--allow-empty-text` | Allows images with missing or empty captions. Also disables the requirement that every image has text. |
| `--non-recursive-data` | Disables recursive dataset scanning. |
| `--crop-mode {center,random,pad}` | Image crop/resize mode. `center` uses center crop, `random` uses random crop, `pad` fits the image into a square canvas. |
| `--horizontal-flip-p FLOAT` | Probability of applying horizontal flip augmentation. |

### Run Location And Resume

| Argument | Description |
| --- | --- |
| `--output PATH` | Output directory for config, checkpoints, and samples. Default: `./runs/pixzig-field`. |
| `--resume PATH` | Path to a `.pt` checkpoint to resume from. |

### Runtime

| Argument | Description |
| --- | --- |
| `--device DEVICE` | Torch device for the model and training tensors, such as `cuda`, `cuda:0`, or `cpu`. |
| `--precision {fp32,fp16,bf16}` | Training autocast precision. `fp32` disables reduced-precision math unless another part of the code changes dtype. |
| `--no-autocast` | Disables autocast even when `--precision` is `fp16` or `bf16`. |
| `--seed INT` | Random seed for Python, NumPy, PyTorch, and dataloader workers. |
| `--num-workers INT` | Number of dataloader workers. |
| `--pin-memory` | Enables DataLoader pinned memory. |

### Image And Batch Shape

| Argument | Description |
| --- | --- |
| `--image-size INT` | Square image size used by the dataset and model. Must be divisible by `--patch-size`. |
| `--patch-size INT` | Non-overlapping patch size used by the model. |
| `--batch-size INT` | Per-step dataloader batch size before gradient accumulation. |
| `--gradient-accumulation-steps INT` | Number of micro-batches accumulated before each optimizer step. |
| `--max-steps INT` | Total optimizer steps to run. |

### Flow And Timestep Sampling

| Argument | Description |
| --- | --- |
| `--timestep-sampling {uniform,logit_normal,cosine,beta,stratified}` | Distribution used to sample flow-matching timesteps. |
| `--timestep-eps FLOAT` | Lower and upper clamp margin for timesteps. |
| `--logit-normal-std FLOAT` | Standard deviation used by `logit_normal` timestep sampling. |
| `--beta-alpha FLOAT` | Alpha parameter used by `beta` timestep sampling. |
| `--beta-beta FLOAT` | Beta parameter used by `beta` timestep sampling. |

### Text Conditioning

| Argument | Description |
| --- | --- |
| `--text-encoder {none,qwen}` | Text-conditioning backend. Use `none` for unconditional training and `qwen` for Qwen embeddings. |
| `--qwen-model-name NAME_OR_PATH` | Hugging Face model name or local Qwen directory. |
| `--allow-qwen-download` | Allows Hugging Face downloads when loading Qwen. By default, Qwen is loaded with local files only. |
| `--qwen-cache-embeddings` | Caches Qwen embeddings in memory by caption text during the run. |
| `--qwen-max-length INT` | Maximum token length passed to the Qwen tokenizer. |
| `--text-encoder-device DEVICE` | Device for the text encoder. Use `model` or `auto` to match the model device. |
| `--cond-dim INT` | Conditioning dimension projected from Qwen embeddings and used by the model. |

### Model Size

| Argument | Description |
| --- | --- |
| `--hidden-dim INT` | Main token width of the PixZig Field model. |
| `--depth INT` | Number of ZigMa backbone blocks. |
| `--zigma-state-dim INT` | State dimension used by the native selective Mamba-style mixer. |
| `--zigma-expand INT` | Expansion factor used inside ZigMa blocks. |
| `--scan-mode TEXT` | Zigzag scan pattern, such as `zigzagN1`, `zigzagN4`, `zigzagN8`, or `zigzagN16`. |
| `--mamba-backend {native,external,auto}` | Selective Mamba backend. `native` uses the built-in PyTorch implementation. |
| `--mamba-inner-expand INT` | Inner expansion factor for the Mamba-style mixer. |
| `--mamba-head-dim INT` | Head dimension for grouped Mamba-style projections. |
| `--mamba-num-groups INT` | Number of groups used by grouped Mamba-style projections. |
| `--mamba-conv-kernel INT` | Depthwise convolution kernel size inside the mixer. |
| `--pixnerd-hidden-dim INT` | Hidden width of the PixNerd-style decoder MLP. |
| `--pixnerd-layers INT` | Number of dynamic PixNerd decoder layers. |
| `--refiner-channels INT` | Channel width of the residual convolutional refiner. |
| `--refiner-blocks INT` | Number of residual refiner blocks. |

### Optimizer And Scheduler

| Argument | Description |
| --- | --- |
| `--learning-rate FLOAT` | AdamW base learning rate. |
| `--weight-decay FLOAT` | AdamW weight decay. |
| `--lr-warmup-steps INT` | Number of warmup steps before cosine decay. |
| `--min-lr-ratio FLOAT` | Minimum learning-rate ratio at the end of cosine decay. |
| `--max-grad-norm FLOAT` | Gradient clipping norm applied before optimizer step. |

### EMA

| Argument | Description |
| --- | --- |
| `--ema-decay FLOAT` | EMA decay value. |
| `--ema-update-every INT` | Number of optimizer steps between EMA updates. |
| `--ema-warmup-steps INT` | Number of steps used by EMA warmup scheduling. |
| `--ema-device DEVICE` | EMA device. Use `model` or `auto` to match the model device, `cpu` to keep EMA off GPU, or `cuda` for GPU EMA. |

### Checkpoints And Safetensors

| Argument | Description |
| --- | --- |
| `--periodic-checkpoints` | Enables periodic checkpoint saving. |
| `--save-every-steps INT` | Saves a periodic checkpoint every N optimizer steps. Also enables periodic checkpoints when provided. |
| `--save-safetensors` | Enables `.safetensors` exports for periodic and final saves. |
| `--safetensors-name NAME` | Final `.safetensors` filename. |
| `--safetensors-source {model,ema}` | Source weights for `.safetensors` export. `ema` exports EMA weights; `model` exports live model weights. |

### Samples During Training

| Argument | Description |
| --- | --- |
| `--periodic-samples` | Enables periodic sample generation during training. |
| `--sample-every-steps INT` | Generates a periodic sample every N optimizer steps. |
| `--sample-n-steps INT` | Number of sampling steps used for training samples. |
| `--sample-method {euler,heun}` | Sampling solver used for training samples. |
| `--sample-prompt TEXT` | Positive sample prompt. |
| `--sample-negative-prompt TEXT` | Negative sample prompt. |
| `--sample-cfg-scale FLOAT` | Classifier-free guidance scale used for samples. |
| `--sample-seed INT` | Random seed used for samples. |
| `--no-sample-ema` | Uses the live model instead of EMA weights for sample generation. |

### Logging

| Argument | Description |
| --- | --- |
| `--log-every-steps INT` | Number of optimizer steps between structured log lines. Live progress is still updated continuously in TTY output. |
