# LoRA Merge & 4-bit Quantization

This folder documents merging a LoRA adapter into a base model and loading the
result in 4-bit precision using `BitsAndBytesConfig` from Hugging Face
Transformers.

Merged model on HF Hub: [atulkrs/opt-mlops-merged](https://huggingface.co/atulkrs/opt-mlops-merged)

---

## What this does

`merge_and_push.py` performs the following steps end-to-end:

1. **Load base model** — `facebook/opt-125m` in fp32 on CPU
2. **Load LoRA adapter** — `atulkrs/opt-mlops-lora` via PEFT
3. **Merge weights** — `PeftModel.merge_and_unload()` bakes adapter deltas into base weights; no adapter overhead at inference
4. **Save locally** — writes merged checkpoint to `./merged-model`
5. **Benchmark** — measures fp32 size, estimates 4-bit size, optionally measures 4-bit load time
6. **Push to HF Hub** — uploads weights, tokenizer, and model card to `atulkrs/opt-mlops-merged`

---

## The merge process

LoRA (Low-Rank Adaptation) adds two small matrices **A** and **B** alongside
each frozen weight matrix **W**. During inference the effective weight is:

```
W_eff = W + α · (B @ A)
```

`merge_and_unload()` computes this sum once and stores the result back in **W**,
then discards A and B entirely. The result is a standard model checkpoint — PEFT
is not required at inference time and there is no runtime overhead from the
adapter.

---

## 4-bit quantization with BitsAndBytesConfig

After merging you can load the model in 4-bit NF4 format to dramatically reduce
memory usage, which is especially useful on consumer GPUs.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 from the QLoRA paper
    bnb_4bit_compute_dtype=torch.float16, # computation dtype (not storage)
)

model = AutoModelForCausalLM.from_pretrained(
    "atulkrs/opt-mlops-merged",
    quantization_config=bnb_config,
    device_map="auto",                   # spread across available GPU/CPU
)
tokenizer = AutoTokenizer.from_pretrained("atulkrs/opt-mlops-merged")
```

### Key parameters explained

| Parameter | Value | Effect |
|-----------|-------|--------|
| `load_in_4bit` | `True` | Activates 4-bit weight quantization via bitsandbytes |
| `bnb_4bit_quant_type` | `"nf4"` | NormalFloat4 — preserves weight distribution better than int4; from QLoRA paper |
| `bnb_4bit_compute_dtype` | `torch.float16` | Dequantizes to fp16 for matrix ops; use `bfloat16` on Ampere+ GPUs |
| `device_map` | `"auto"` | Automatically places layers on GPU/CPU based on available VRAM |

### Why NF4 over INT4?

Standard INT4 assumes weights are uniformly distributed. NF4 uses quantile
normalization so that each quantization bin covers an equal number of weight
values, preserving the bell-curve distribution that neural network weights
typically follow. This gives lower quantization error at the same bit-width.

---

## Size benchmark

Measured on `facebook/opt-125m` with `atulkrs/opt-mlops-lora` merged in:

| Format | Size | Method |
|--------|------|--------|
| FP32 (merged) | **477.8 MB** | measured — `sum(p.numel() * p.element_size())` |
| 4-bit NF4 (estimated) | **~59.7 MB** | ≈ fp32 ÷ 8 (2 params/byte + scale overhead) |
| **Reduction** | **~8×** | |

> The exact 4-bit size depends on bitsandbytes quantization block size (default 64)
> and the scale/zero-point metadata, so real measured size will differ slightly
> from the fp32 ÷ 8 estimate.

---

## Running the script

```bash
# Install dependencies
pip install torch transformers peft huggingface_hub accelerate

# Set token (write access required to push to Hub)
export HF_TOKEN=hf_...          # Linux/macOS
$env:HF_TOKEN = "hf_..."        # PowerShell

# Run
python merge_and_push.py
```

The script reads `HF_TOKEN` from the environment — never hardcode tokens in
source files.

### Output

```
INFO Loading base model: facebook/opt-125m
INFO Base model loaded in 108.8s
INFO Loading LoRA adapter: atulkrs/opt-mlops-lora
INFO Merging adapter weights into base model …
INFO Merge complete in 0.0s
INFO Saving merged model to merged-model …
INFO Saved.
── Model Size Benchmark ──────────────────────────────────
  FP32 merged        :    477.8 MB
  Estimated 4-bit    :     59.7 MB  (≈ fp32 / 8)
  Size reduction     :     ~8.0x smaller
──────────────────────────────────────────────────────────
INFO Pushing merged model to: atulkrs/opt-mlops-merged
✅ Done!  https://huggingface.co/atulkrs/opt-mlops-merged
```

---

## Requirements

```
torch>=2.0
transformers>=4.40
peft>=0.9
huggingface_hub>=0.20
accelerate>=0.27      # required for device_map="auto"
bitsandbytes>=0.46.1  # Linux + CUDA only; needed for 4-bit loading
```

> `bitsandbytes` requires a Linux environment with CUDA. The merge and push
> steps in `merge_and_push.py` run on CPU and work on any OS.
