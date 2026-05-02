"""
GPU monitor — records per-sample GPU metrics during training.
Run in a separate terminal:
    python gpu_monitor.py --out gpu_log.csv --interval 0.5

Then plot with:
    python gpu_monitor.py --plot gpu_log.csv
"""
import argparse
import csv
import time
import os
import signal
import sys

import threading

try:
    import pynvml
    PYNVML = True
except ImportError:
    PYNVML = False

FIELDS = ["time_s", "sm_util_%", "mem_util_%", "gpu_mem_used_mb", "gpu_mem_total_mb",
          "power_w", "temp_c", "clock_sm_mhz", "clock_mem_mhz"]


class GPUMonitor:
    """Embeddable GPU monitor — runs in a background thread."""

    def __init__(self, out_path: str, interval: float = 0.5, gpu_idx: int = 0):
        self.out_path = out_path
        self.interval = interval
        self.gpu_idx = gpu_idx
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not PYNVML:
            print("[GPUMonitor] pynvml not available, skipping.")
            return
        pynvml.nvmlInit()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[GPUMonitor] Logging to {self.out_path}")

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        pynvml.nvmlShutdown()
        print(f"[GPUMonitor] Done → {self.out_path}")

    def _run(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_idx)
        with open(self.out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time_s", "sm_util_%", "mem_util_%",
                                                    "gpu_mem_used_mb", "power_w", "temp_c", "clock_sm_mhz"])
            writer.writeheader()
            while not self._stop.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except: power = float("nan")
                try:    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except: temp = float("nan")
                try:    clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                except: clk = float("nan")
                writer.writerow({
                    "time_s":          round(time.time() - self._t0, 2),
                    "sm_util_%":       util.gpu,
                    "mem_util_%":      util.memory,
                    "gpu_mem_used_mb": round(mem.used / 1024**2, 1),
                    "power_w":         round(power, 1),
                    "temp_c":          temp,
                    "clock_sm_mhz":    clk,
                })
                f.flush()
                self._stop.wait(self.interval)


def plot(csv_path: str, smoothing: int = 10):
    import csv as csv_mod
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    rows = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})

    if not rows:
        print("No data.")
        return

    def col(key): return [r[key] for r in rows]
    def smooth(xs):
        if smoothing <= 1:
            return xs
        import statistics
        out = []
        for i in range(len(xs)):
            window = xs[max(0, i - smoothing):i + smoothing + 1]
            out.append(statistics.mean(window))
        return out

    t = col("time_s")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"GPU Profile — {os.path.basename(csv_path)}", fontsize=13)

    # 1. SM + Memory utilisation
    ax = axes[0]
    ax.plot(t, smooth(col("sm_util_%")),  label="SM util %",  color="tab:blue")
    ax.plot(t, smooth(col("mem_util_%")), label="Mem BW util %", color="tab:orange", alpha=0.7)
    ax.set_ylabel("Utilisation %")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="red", linewidth=0.5, linestyle="--")
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))

    # 2. GPU memory used
    ax = axes[1]
    ax.plot(t, col("gpu_mem_used_mb"), color="tab:green")
    ax.set_ylabel("VRAM used (MB)")
    total = None
    if PYNVML:
        try:
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            total = pynvml.nvmlDeviceGetMemoryInfo(h).total / 1024**2
            pynvml.nvmlShutdown()
        except Exception:
            pass
    if total:
        ax.set_ylim(0, total * 1.05)
        ax.axhline(total, color="red", linewidth=0.5, linestyle="--", label=f"Total {total:.0f} MB")
        ax.legend(loc="upper right", fontsize=8)

    # 3. Power
    ax = axes[2]
    ax.plot(t, smooth(col("power_w")), color="tab:red")
    ax.set_ylabel("Power (W)")

    # 4. SM clock
    ax = axes[3]
    ax.plot(t, smooth(col("clock_sm_mhz")), color="tab:purple", label="SM clock")
    ax.set_ylabel("Clock (MHz)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = csv_path.replace(".csv", "_plot.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",      default="gpu_log.csv", help="output CSV path")
    parser.add_argument("--interval", default=0.5, type=float, help="sample interval in seconds")
    parser.add_argument("--gpu",      default=0, type=int, help="GPU index")
    parser.add_argument("--plot",     default=None, metavar="CSV", help="plot an existing CSV instead of monitoring")
    parser.add_argument("--smooth",   default=10, type=int, help="smoothing window for plot")
    args = parser.parse_args()

    if args.plot:
        plot(args.plot, smoothing=args.smooth)
    else:
        monitor = GPUMonitor(out_path=args.out, interval=args.interval, gpu_idx=args.gpu)
        monitor.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop()
