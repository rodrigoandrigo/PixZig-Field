#! /bin/bash

QWEN_MODEL="${QWEN_MODEL:-./Qwen3.5-2B-Base}"
TEXT_ENCODER_DEVICE="${TEXT_ENCODER_DEVICE:-cuda}"
EMA_DEVICE="${EMA_DEVICE:-cpu}"
SEED="${SEED:-42}"
USE_QWEN="${USE_QWEN:-0}"

TEXT_ARGS=(--text-encoder none --allow-empty-text)
if [[ "$USE_QWEN" == "1" ]]; then
  TEXT_ARGS=(
    --text-encoder qwen
    --qwen-model-name "$QWEN_MODEL"
    --qwen-cache-embeddings
    --text-encoder-device "$TEXT_ENCODER_DEVICE"
    --cond-dim 1024
  )
fi

python3 -m pixzig_field.train \
  --data "./dataset" \
  --output "./runs/small-256" \
  --device cuda \
  --image-size 256 \
  --patch-size 16 \
  --batch-size 1 \
  --num-workers 2 \
  --pin-memory \
  --max-steps 1280 \
  --gradient-accumulation-steps 1 \
  --crop-mode pad \
  --horizontal-flip-p 0.0 \
  --timestep-sampling uniform \
  --precision "fp32" \
  "${TEXT_ARGS[@]}" \
  --hidden-dim 256 \
  --depth 2 \
  --zigma-state-dim 16 \
  --zigma-expand 1 \
  --scan-mode zigzagN4 \
  --mamba-backend native \
  --mamba-inner-expand 1 \
  --mamba-head-dim 64 \
  --mamba-num-groups 1 \
  --mamba-conv-kernel 4 \
  --pixnerd-hidden-dim 64 \
  --pixnerd-layers 1 \
  --refiner-channels 16 \
  --refiner-blocks 1 \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --lr-warmup-steps 25 \
  --min-lr-ratio 0.1 \
  --max-grad-norm 1.0 \
  --ema-decay 0.999 \
  --ema-update-every 1 \
  --ema-warmup-steps 0 \
  --ema-device "$EMA_DEVICE" \
  --periodic-checkpoints \
  --save-every-steps 128 \
  --save-safetensors \
  --safetensors-name pixzig-small-256.safetensors \
  --safetensors-source ema \
  --periodic-samples \
  --sample-every-steps 128 \
  --sample-n-steps 8 \
  --sample-method heun \
  --sample-prompt "a clean image" \
  --sample-negative-prompt "" \
  --sample-cfg-scale 1.0 \
  --sample-seed 1234 \
  --no-sample-ema \
  --seed "$SEED" \
  --log-every-steps 10
