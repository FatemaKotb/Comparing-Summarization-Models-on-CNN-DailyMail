import torch


def should_log(i: int, total: int, every: int) -> bool:
    """Return True every N steps, and also on the final step."""
    return (i % every == 0) or (i == total)


def fmt_cuda_mem() -> str:
    """Human-readable CUDA memory usage (allocated/reserved/peak)."""
    if not torch.cuda.is_available():
        return "cuda: not available"
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserv = torch.cuda.memory_reserved() / (1024 ** 3)
    max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return f"cuda mem | alloc={alloc:.2f}GB reserved={reserv:.2f}GB max_alloc={max_alloc:.2f}GB"
