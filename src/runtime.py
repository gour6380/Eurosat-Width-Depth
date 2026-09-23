"""Backend selection and lean per-batch operations; importing starts no work."""

import os
import random

import numpy as np
import psutil
import torch


def select_device():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for an explicit backend experiment")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        # Also applies in a fresh kernel that only reuses/evaluates completed fits.
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return torch.device("cuda")
    return torch.device("cpu")


def device_information(run):
    return {
        "device": str(run.device),
        "precision": "float32",
        "gpu_name": torch.cuda.get_device_name(run.device)
        if run.device.type == "cuda"
        else ("Apple Metal (MPS)" if run.device.type == "mps" else "CPU"),
        "memory_definition": "Snapshots; process RSS and accelerator allocations are not additive",
        **memory_snapshot(run.device),
    }


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def capture_rng():
    result = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def memory_snapshot(device):
    values = {"process_rss_mib": psutil.Process().memory_info().rss / 2**20}
    if device.type == "mps":
        values.update(
            mps_live_mib=torch.mps.current_allocated_memory() / 2**20,
            mps_driver_mib=torch.mps.driver_allocated_memory() / 2**20,
        )
    elif device.type == "cuda":
        values.update(
            cuda_allocated_mib=torch.cuda.memory_allocated(device) / 2**20,
            cuda_reserved_mib=torch.cuda.memory_reserved(device) / 2**20,
        )
    return values


def prepare_images(images, data, device, augment=False):
    """Transfer one batch, augment on device, then apply training-only RGB statistics.

    augment is False or the TrainingConfig (avoids hidden augmentation defaults).
    No per-image CPU/GPU transfers or scalar reads occur here.
    """
    x = images.to(device=device, dtype=torch.float32).div_(255)
    if augment:
        if augment.horizontal_flip:
            mask = torch.rand((len(x), 1, 1, 1), device=device) < 0.5
            x = torch.where(mask, x.flip(-1), x)
        if augment.vertical_flip:
            mask = torch.rand((len(x), 1, 1, 1), device=device) < 0.5
            x = torch.where(mask, x.flip(-2), x)
    # Tiny statistics tensors cached by backend to avoid repeat host transfers.
    cache = getattr(data, "_normalization_cache", None)
    if cache is None:
        cache = {}
        data._normalization_cache = cache
    key = str(device)
    if key not in cache:
        cache[key] = (
            torch.tensor(data.mean, device=device, dtype=torch.float32).view(1, 3, 1, 1),
            torch.tensor(data.std, device=device, dtype=torch.float32).view(1, 3, 1, 1),
        )
    mean, std = cache[key]
    return x.sub_(mean).div_(std)
