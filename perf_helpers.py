"""Helper functions for the anemoi performance profiling notebook.

You do not have to read this file to do the exercises.
The functions start anemoi profiling runs and read the files that the runs make.
"""

import pickle
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs")
CONFIG_NAME = "perf_profiling"


def run_experiment(name: str, overrides: list[str] | None = None, force: bool = False) -> Path:
    """Run one anemoi profiling job and return its profiler folder.

    Parameters
    ----------
    name : str
        Short name for the experiment. The output goes into ``outputs/<name>``.
    overrides : list[str], optional
        Hydra overrides, for example ``["model.processor.num_chunks=8"]``.
    force : bool, optional
        If True, run the job again when results for this name exist.

    Returns
    -------
    Path
        The folder that contains the profiler files.
    """
    overrides = overrides or []
    root = OUTPUT_DIR / name
    existing = find_profiler_dir(name)
    if existing is not None and not force:
        print(f"Results for '{name}' exist. Use force=True to run again.")
        return existing

    root.mkdir(parents=True, exist_ok=True)
    log_file = root / "run.log"
    cmd = [
        "anemoi-training",
        "profile",
        f"--config-name={CONFIG_NAME}",
        f"system.output.root={root}",
        *overrides,
    ]
    print("Command:", " ".join(cmd))
    print(f"Log file: {log_file}")
    start = time.time()
    with log_file.open("w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    print(f"Finished in {time.time() - start:.0f} s with exit code {result.returncode}.")

    if result.returncode != 0:
        print("The run failed. These are the last lines of the log:\n")
        print("\n".join(log_file.read_text().splitlines()[-25:]))
        if "OutOfMemoryError" in log_file.read_text():
            print("\n>>> The GPU did not have sufficient memory (CUDA out of memory).")
        return None
    return find_profiler_dir(name)


def find_profiler_dir(name: str) -> Path | None:
    """Return the newest profiler folder for an experiment, or None."""
    candidates = sorted((OUTPUT_DIR / name / "profiler").glob("*/"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def find_trace_file(profiler_dir: Path) -> Path | None:
    """Return the PyTorch profiler trace of the training steps in a profiler folder.

    The profiler can write more than one ``*.pt.trace.json`` file (one for training, one for validation).
    The training trace is the first file, thus we select the oldest file.
    """
    traces = sorted(Path(profiler_dir).glob("*.pt.trace.json"), key=lambda p: p.stat().st_mtime)
    return traces[0] if traces else None


def load_snapshot(profiler_dir: Path) -> dict:
    """Load the memory snapshot (``memory_snapshot.pickle``) of a run."""
    with (Path(profiler_dir) / "memory_snapshot.pickle").open("rb") as f:
        return pickle.load(f)


def memory_timeline(snapshot: dict, device: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the allocated GPU memory over time from a memory snapshot.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Time in seconds from the start of the recording, and allocated memory in GB.
    """
    events = [e for e in snapshot["device_traces"][device] if e["action"] in ("alloc", "free_completed")]
    sign = np.array([1 if e["action"] == "alloc" else -1 for e in events])
    sizes = np.array([e["size"] for e in events]) * sign
    times = np.array([e["time_us"] for e in events], dtype=float)
    memory = np.cumsum(sizes)

    # The trace only has the events after the recording started.
    # Memory that was allocated before the start (for example the model weights) is not in the trace.
    # We add it back, so that the last value is equal to the memory in use when the snapshot was saved.
    in_use_at_end = sum(
        b["size"]
        for seg in snapshot["segments"]
        if seg["device"] == device
        for b in seg["blocks"]
        if b["state"] == "active_allocated"
    )
    memory = memory + (in_use_at_end - memory[-1])
    return (times - times[0]) / 1e6, memory / 1e9


def peak_memory_gb(profiler_dir: Path) -> float:
    """Return the peak allocated GPU memory (GB) during the recorded steps."""
    _, memory = memory_timeline(load_snapshot(profiler_dir))
    return float(memory.max())


def training_throughput(profiler_dir: Path) -> float:
    """Return the average training throughput (iterations per second) from the speed report."""
    df = pd.read_csv(Path(profiler_dir) / "speed_profiler.csv", index_col=0)
    return float(df.set_index("metric").loc["training_avg_throughput", "value"])


def plot_memory_timelines(experiments: dict[str, Path], title: str = "GPU memory during training steps") -> None:
    """Plot the memory timeline of one or more experiments on the same axes."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, profiler_dir in experiments.items():
        if profiler_dir is None:
            continue
        t, mem = memory_timeline(load_snapshot(profiler_dir))
        ax.plot(t, mem, label=f"{name} (peak {mem.max():.1f} GB)", linewidth=1.2)
    ax.set_xlabel("Time from start of recording [s]")
    ax.set_ylabel("Allocated GPU memory [GB]")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()


def summary_table(experiments: dict[str, Path]) -> pd.DataFrame:
    """Make a table with peak memory and throughput for each experiment."""
    rows = []
    for name, profiler_dir in experiments.items():
        if profiler_dir is None:
            rows.append({"experiment": name, "peak memory [GB]": np.nan, "throughput [it/s]": np.nan})
            continue
        rows.append(
            {
                "experiment": name,
                "peak memory [GB]": round(peak_memory_gb(profiler_dir), 2),
                "throughput [it/s]": round(training_throughput(profiler_dir), 2),
            },
        )
    return pd.DataFrame(rows).set_index("experiment")
