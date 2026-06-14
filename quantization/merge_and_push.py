"""
merge_and_push.py

1. Loads facebook/opt-125m (base)
2. Loads LoRA adapter atulkrs/opt-mlops-lora via PEFT
3. Merges weights with merge_and_unload()
4. Saves merged model to ./merged-model
5. Benchmarks fp32 vs estimated 4-bit size; measures 4-bit load time if possible
6. Pushes merged model + model card to atulkrs/opt-mlops-merged on HF Hub
"""

import os
import sys
import time
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_MODEL_ID  = "facebook/opt-125m"
ADAPTER_ID     = "atulkrs/opt-mlops-lora"
MERGED_REPO_ID = "atulkrs/opt-mlops-merged"
LOCAL_DIR      = Path("./merged-model")

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    log.error("HF_TOKEN not set. Export HF_TOKEN=hf_... before running.")
    sys.exit(1)


# ── 1. Base model ─────────────────────────────────────────────────────────────
log.info(f"Loading base model: {BASE_MODEL_ID}")
t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
log.info(f"Base model loaded in {time.time() - t0:.1f}s")


# ── 2. LoRA adapter ───────────────────────────────────────────────────────────
log.info(f"Loading LoRA adapter: {ADAPTER_ID}")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_ID, token=HF_TOKEN)


# ── 3. Merge and unload ───────────────────────────────────────────────────────
log.info("Merging adapter weights into base model …")
t_merge = time.time()
merged_model = peft_model.merge_and_unload()
log.info(f"Merge complete in {time.time() - t_merge:.1f}s")


# ── 4. Save locally ───────────────────────────────────────────────────────────
log.info(f"Saving merged model to {LOCAL_DIR} …")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(str(LOCAL_DIR))
tokenizer.save_pretrained(str(LOCAL_DIR))
log.info("Saved.")


# ── Benchmark ─────────────────────────────────────────────────────────────────
def param_size_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

fp32_mb      = param_size_mb(merged_model)
est_4bit_mb  = fp32_mb / 8.0          # NF4 packs 2 params/byte; rough upper bound

log.info("\n── Model Size Benchmark ──────────────────────────────────")
log.info(f"  FP32 merged        : {fp32_mb:>8.1f} MB")
log.info(f"  Estimated 4-bit    : {est_4bit_mb:>8.1f} MB  (≈ fp32 / 8)")
log.info(f"  Size reduction     :     ~{fp32_mb / est_4bit_mb:.1f}x smaller")
log.info("──────────────────────────────────────────────────────────\n")

actual_4bit_mb = None
load_4bit_s    = None
try:
    log.info("Measuring 4-bit load time from local checkpoint …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    t_4bit = time.time()
    model_4bit = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
    )
    load_4bit_s    = time.time() - t_4bit
    actual_4bit_mb = param_size_mb(model_4bit)
    log.info(f"  4-bit load time    : {load_4bit_s:.1f}s")
    log.info(f"  4-bit actual size  : {actual_4bit_mb:.1f} MB")
    del model_4bit
except Exception as exc:
    log.warning(f"  4-bit benchmark skipped: {exc}")


# ── 5. Push model to Hub ──────────────────────────────────────────────────────
log.info(f"Pushing merged model to: {MERGED_REPO_ID}")
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=MERGED_REPO_ID, repo_type="model", exist_ok=True, private=False)

merged_model.push_to_hub(MERGED_REPO_ID, token=HF_TOKEN)
tokenizer.push_to_hub(MERGED_REPO_ID, token=HF_TOKEN)


# ── 6. Model card ─────────────────────────────────────────────────────────────
size_table_rows = (
    f"| FP32 (merged)    | {fp32_mb:.1f} MB         | measured              |\n"
    f"| 4-bit NF4 (est.) | {est_4bit_mb:.1f} MB          | ≈ fp32 / 8            |"
)
if actual_4bit_mb is not None:
    size_table_rows += (
        f"\n| 4-bit NF4 (actual)| {actual_4bit_mb:.1f} MB         | measured              |"
    )

load_time_note = (
    f"4-bit load time measured from local checkpoint: **{load_4bit_s:.1f}s**"
    if load_4bit_s is not None
    else "4-bit load time benchmark not available (bitsandbytes requires Linux + CUDA)."
)

MODEL_CARD = f"""\
---
language: en
license: mit
base_model: {BASE_MODEL_ID}
tags:
  - opt
  - lora
  - peft
  - merged
  - mlops
  - causal-lm
---

# opt-mlops-merged

**facebook/opt-125m** fine-tuned with a LoRA adapter
([atulkrs/opt-mlops-lora](https://huggingface.co/atulkrs/opt-mlops-lora))
and fully merged into base weights via `PeftModel.merge_and_unload()`.

The adapter deltas are baked in — no PEFT dependency needed at inference time.

## Load (full precision)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("{MERGED_REPO_ID}")
tokenizer = AutoTokenizer.from_pretrained("{MERGED_REPO_ID}")
```

## Load in 4-bit with BitsAndBytes (recommended for GPU inference)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — from QLoRA paper
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "{MERGED_REPO_ID}",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{MERGED_REPO_ID}")
```

> **Tip:** swap `torch.float16` for `torch.bfloat16` on Ampere+ GPUs (A100, RTX 30xx+)
> for better numerical stability at no speed cost.

## Size & load-time benchmark

| Format            | Size                | Source                |
|-------------------|---------------------|-----------------------|
{size_table_rows}

{load_time_note}

## Merge details

| Field          | Value                                      |
|----------------|--------------------------------------------|
| Base model     | facebook/opt-125m                          |
| Adapter        | atulkrs/opt-mlops-lora                     |
| Merge method   | `PeftModel.merge_and_unload()`             |
| Saved format   | SafeTensors (fp32)                         |

### Why merge?

Merging removes the adapter overhead entirely — no extra matrix multiplications at
inference, no PEFT dependency, and the weights load like any standard
`transformers` checkpoint. The only trade-off is that you can no longer swap
adapters without re-loading the base model.
"""

api.upload_file(
    path_or_fileobj=MODEL_CARD.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=MERGED_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
    commit_message="Add model card with 4-bit loading instructions and size benchmark",
)

log.info(f"\n✅ Done!  https://huggingface.co/{MERGED_REPO_ID}")
