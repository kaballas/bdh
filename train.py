# Copyrighth Pathway Technology, Inc.

import os
from contextlib import nullcontext

import bdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
CHECKPOINT_PATH = os.getenv(
    "BDH_CHECKPOINT_PATH", os.path.join("checkpoints", "bdh_checkpoint.pt")
)
CHECKPOINT_EVERY = int(os.getenv("BDH_CHECKPOINT_EVERY", str(LOG_FREQ)))

input_file_path = "/content/bdh/simpleqa_verified_extract.txt"


def get_batch(split):
    # treat the file as bytes
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()


def save_checkpoint(path, step, model, optimizer, scaler):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)
    print(f"Saved checkpoint to '{path}' at step {step}")


def load_checkpoint(path, model, optimizer, scaler):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler.is_enabled() and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    # Ensure optimizer tensors live on the right device after loading
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    return int(checkpoint.get("step", -1))


if __name__ == "__main__":

    model = bdh.BDH(BDH_CONFIG).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    start_step = 0
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint_step = load_checkpoint(
                CHECKPOINT_PATH, model, optimizer, scaler
            )
        except Exception as exc:
            print(
                f"Failed to load checkpoint '{CHECKPOINT_PATH}': {exc}."
                " Starting from scratch."
            )
            checkpoint_step = -1
        start_step = max(0, checkpoint_step + 1)
        if checkpoint_step >= 0:
            print(
                f"Loaded checkpoint from '{CHECKPOINT_PATH}' at step {checkpoint_step}."
            )
            if start_step >= MAX_ITERS:
                print("Checkpoint already reached MAX_ITERS; skipping further training.")

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0
    for step in range(start_step, MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)
        x, y = get_batch("train")
        loss_acc += loss
        loss_steps += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if step % LOG_FREQ == 0:
            print(f"Step: {step}/{MAX_ITERS} loss {loss_acc.item() / loss_steps:.3}")
            loss_acc = 0
            loss_steps = 0
        if CHECKPOINT_EVERY > 0 and ((step + 1) % CHECKPOINT_EVERY == 0 or step == MAX_ITERS - 1):
            save_checkpoint(CHECKPOINT_PATH, step, model, optimizer, scaler)
    print("Training done, now generating a sample ")
    model.eval()
    prompt = torch.tensor(
        bytearray("The letter **a** is ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
