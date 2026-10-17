"""Optional forward-only full/induced attention diagnostic; never run by pytest.

Run with uv run --locked python scripts/diagnose_attention.py --help.
CPU memory is positive allocation traffic in one profiled forward, not peak
resident memory. CUDA memory is incremental peak allocated tensor memory above
warmed-up model/input storage. Timing excludes profiling, construction, and input
transfer. Results describe untrained eval-mode models, not PPO training memory.
"""

import argparse
import csv
import json
import statistics
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

from geogb.models import make_policy
from geogb.observations import PairBatch


def positive_int(value: str) -> int:
    """Parse a positive CLI integer."""
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous CUDA kernels before reading time or memory."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model: torch.nn.Module, batch: PairBatch, *, warmup: int,
            repeats: int) -> tuple[float, int | None, int | None]:
    """Return median milliseconds, CPU allocation bytes, CUDA peak delta bytes."""
    device = batch.features.device
    samples = []
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        synchronize(device)
        for _ in range(repeats):
            start = time.perf_counter()
            model(batch)
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            model(batch)
            synchronize(device)
            cpu_bytes = None
            cuda_bytes = torch.cuda.max_memory_allocated(device) - baseline
        else:
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as profiler:
                model(batch)
            cpu_bytes = sum(max(0, event.self_cpu_memory_usage)
                            for event in profiler.events())
            cuda_bytes = None
    return statistics.median(samples), cpu_bytes, cuda_bytes


def main() -> None:
    """Compare otherwise matching SAB and ISAB configurations across set sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--feature-dim", type=positive_int, default=12)
    parser.add_argument("--embed-dim", type=positive_int, default=64)
    parser.add_argument("--num-heads", type=positive_int, default=4)
    parser.add_argument("--depth", type=positive_int, default=2)
    parser.add_argument("--feedforward-dim", type=positive_int, default=128)
    parser.add_argument("--num-inducing", type=positive_int, default=16)
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.embed_dim % args.num_heads:
        parser.error("embed-dim must be divisible by num-heads")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    print(json.dumps({**vars(args), "torch": torch.__version__,
                      "cuda_runtime": torch.version.cuda,
                      "device_name": (torch.cuda.get_device_name(device)
                                      if device.type == "cuda" else "cpu")}), file=sys.stderr)
    writer = csv.writer(sys.stdout)
    writer.writerow(["policy", "pairs", "median_forward_ms", "cpu_allocation_bytes",
                     "cuda_incremental_peak_bytes", "attention_interactions"])
    for size in args.sizes:
        batch = PairBatch(torch.rand(args.batch_size, size, args.feature_dim, device=device),
                          torch.ones(args.batch_size, size, dtype=torch.bool, device=device))
        for name in ["set_transformer", "induced_set_transformer"]:
            kwargs = dict(embed_dim=args.embed_dim, num_heads=args.num_heads,
                          depth=args.depth, feedforward_dim=args.feedforward_dim, dropout=0)
            if name == "induced_set_transformer":
                kwargs["num_inducing"] = args.num_inducing
            model = make_policy(name, args.feature_dim, **kwargs).to(device).eval()
            elapsed, cpu_bytes, cuda_bytes = measure(model, batch, warmup=args.warmup,
                                                     repeats=args.repeats)
            interactions = (size * size if name == "set_transformer"
                            else 2 * size * args.num_inducing)
            interactions *= args.batch_size * args.num_heads * args.depth
            writer.writerow([name, size, f"{elapsed:.6f}", cpu_bytes, cuda_bytes, interactions])
            del model


if __name__ == "__main__":
    main()
