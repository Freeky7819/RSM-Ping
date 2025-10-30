
#!/usr/bin/env python3
# RSM-Ping (Open Edition) — Resonant Stillness Monitor
# License: Apache-2.0 © Freedom (Damjan) 2025
#
# Harmonic Signature Protocol
# {
#   "omega": 6.0,
#   "phi": 1.047,
#   "gamma": 0.0,
#   "intent": "resonant_network_stillness",
#   "author": "Freedom (Damjan)"
# }

import argparse, os, platform, re, subprocess, sys, math, json
from datetime import datetime
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def parse_args():
    p = argparse.ArgumentParser(description="RSM-Ping — Resonant Stillness Monitor (Open)")
    p.add_argument("--host", type=str, default="8.8.8.8", help="Host/IP to ping")
    p.add_argument("--count", type=int, default=150, help="Number of samples")
    p.add_argument("--interval", type=float, default=0.5, help="Interval seconds (ping or synthetic)")
    p.add_argument("--threshold", type=float, default=0.01, help="Stillness threshold for jitter variance (ms^2)")
    p.add_argument("--adaptive", type=int, default=1, help="Adaptive threshold (1=yes,0=no)")
    p.add_argument("--factor", type=float, default=1.5, help="Adaptive threshold factor")
    p.add_argument("--mode", type=str, default="auto", choices=["auto","ping","synthetic"], help="Acquisition mode")
    p.add_argument("--simulate", type=str, default="none", choices=["none","resonant","chaotic"], help="Synthetic pattern")
    p.add_argument("--plot", type=int, default=1, help="Save 4-panel plot (1=yes,0=no)")
    p.add_argument("--outdir", type=str, default="results", help="Output directory")
    p.add_argument("--csv", type=str, default=None, help="CSV filename (auto if not set)")
    p.add_argument("--verbose", type=int, default=0, help="Print per-sample RTTs")
    return p.parse_args()

def ping_once(host: str):
    system = platform.system().lower()
    cmd = ["ping", "-n", "1", host] if system == "windows" else ["ping", "-c", "1", host]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    except Exception:
        return None
    m = re.search(r"time[=<]\s*(\d+(?:\.\d+)?)\s*ms", proc.stdout, flags=re.IGNORECASE)
    if not m: return None
    try: return float(m.group(1))
    except Exception: return None

def run_ping_batch(host: str, count: int, interval: float):
    system = platform.system().lower()
    if system == "windows":
        cmd = ["ping", "-n", str(count), host]
    else:
        cmd = ["ping", "-c", str(count)]
        if interval > 0: cmd += ["-i", str(interval)]
        cmd += [host]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print("ERROR: 'ping' command not found.", file=sys.stderr)
        return []
    pattern = re.compile(r"time[=<]\s*(\d+(?:\.\d+)?)\s*ms", re.IGNORECASE)
    rtts = []
    for line in proc.stdout.splitlines():
        m = pattern.search(line)
        if m:
            try: rtts.append(float(m.group(1)))
            except Exception: pass
    return rtts

def synth_series(n, mode="none", base=20.0):
    import numpy as _np, math as _m
    if mode == "resonant":
        return [_np.float64(base + _m.sin(i/5.0)*0.5 + _np.random.normal(0, 0.2)) for i in range(n)]
    if mode == "chaotic":
        return [_np.float64(base + _np.random.normal(0, 3.0)) for _ in range(n)]
    return [_np.float64(base + _np.random.normal(0, 1.0)) for _ in range(n)]

def compute_resonant_stillness(rtts, threshold, adaptive=False, factor=1.5):
    if len(rtts) < 2:
        return {"avg_latency_ms": float("nan"), "jitter_var_ms2": float("nan"),
                "stillness": 0.0, "spectral_entropy": float("nan"),
                "autocorr_decay": float("nan"), "resonant_score": 0.0,
                "samples": len(rtts), "jitter_series": []}
    rtts = np.array(rtts, dtype=float)
    diffs = np.diff(rtts)
    jitter_var = float(np.var(diffs))
    avg_latency = float(np.mean(rtts))

    thr = float(threshold) if threshold > 0 else 0.0
    if adaptive and len(diffs) >= 8:
        wins = [np.var(diffs[i-4:i+1]) for i in range(4, len(diffs))]
        if len(wins) > 0:
            baseline_var = float(np.percentile(wins, 25))
            thr = max(1e-12, baseline_var * float(factor))

    S = float(math.exp(-jitter_var / thr)) if thr > 0 else 0.0

    fft_vals = np.abs(np.fft.rfft(diffs))
    psd = fft_vals ** 2
    s = float(np.sum(psd))
    if s <= 0:
        H = 0.0
    else:
        p = psd / s
        H = float(-np.sum(p * np.log(p + 1e-12)))
        H /= float(np.log(len(p) + 1e-12))

    diffs_z = diffs - np.mean(diffs)
    ac_full = np.correlate(diffs_z, diffs_z, mode='full')
    ac = ac_full[ac_full.size // 2:]
    thresh = ac[0] * 0.1 if ac[0] > 0 else 0
    di = 0
    for i in range(len(ac)):
        if ac[i] < thresh:
            di = i
            break
    D = float(di / max(1, len(ac)))

    inv_e = 1.0 - max(0.0, min(1.0, H))
    inv_d = 1.0 - max(0.0, min(1.0, D))
    R = float((max(1e-6, S) * max(1e-6, inv_e) * max(1e-6, inv_d)) ** (1/3))

    return {
        "avg_latency_ms": avg_latency,
        "jitter_var_ms2": jitter_var,
        "stillness": S,
        "spectral_entropy": H,
        "autocorr_decay": D,
        "resonant_score": R,
        "samples": len(rtts),
        "jitter_series": diffs.tolist(),
        "adaptive_threshold": thr,
    }

def save_csv(outdir, csv_name, host, rtts, result, threshold):
    os.makedirs(outdir, exist_ok=True)
    if not csv_name:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_name = f"rsm_ping_{host.replace('.', '_')}_{ts}.csv"
    csv_path = os.path.join(outdir, csv_name)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,rtt_ms\n")
        for i, r in enumerate(rtts):
            f.write(f"{i},{float(r):.3f}\n")
        f.write("\n")
        for k in ["avg_latency_ms","jitter_var_ms2","stillness","spectral_entropy","autocorr_decay","resonant_score","adaptive_threshold"]:
            v = result.get(k, float("nan"))
            f.write(f"{k},{v}\n")
        f.write(f"threshold_ms2,{threshold}\n")
    return csv_path

def save_plot(outdir, host, rtts, result):
    if plt is None or len(rtts) < 6:
        return None
    import matplotlib.pyplot as _plt
    rtts_arr = np.array(rtts, dtype=float)
    diffs = np.diff(rtts_arr)
    win = 8
    def rolling_var(x, w):
        return np.array([np.var(x[i-w+1:i+1]) for i in range(w-1, len(x))])
    def rolling_entropy(x, w):
        out = []
        for i in range(w-1, len(x)):
            seg = x[i-w+1:i+1]
            fft_vals = np.abs(np.fft.rfft(seg))
            psd = fft_vals**2
            s = psd.sum()
            if s <= 0: out.append(0.0)
            else:
                p = psd / s
                H = -np.sum(p * np.log(p + 1e-12))
                H /= np.log(len(p) + 1e-12)
                out.append(H)
        return np.array(out)
    def rolling_autocorr_decay(x, w):
        out = []
        for i in range(w-1, len(x)):
            seg = x[i-w+1:i+1] - np.mean(x[i-w+1:i+1])
            ac_full = np.correlate(seg, seg, mode="full")
            ac = ac_full[ac_full.size//2:]
            thresh = ac[0]*0.1 if ac[0] > 0 else 0
            di = 0
            for k in range(len(ac)):
                if ac[k] < thresh:
                    di = k
                    break
            out.append(di / max(1, len(ac)))
        return np.array(out)

    jitter_var_roll = rolling_var(diffs, win)
    thr = result.get("adaptive_threshold", 0.01)
    S_roll = np.exp(-np.clip(jitter_var_roll, 0, None) / max(thr, 1e-12))
    H_roll = rolling_entropy(diffs, win)
    D_roll = rolling_autocorr_decay(diffs, win)
    inv_e = 1.0 - np.clip(H_roll, 0, 1)
    inv_d = 1.0 - np.clip(D_roll, 0, 1)
    R_roll = (np.clip(S_roll, 1e-6, 1.0) * np.clip(inv_e, 1e-6, 1.0) * np.clip(inv_d, 1e-6, 1.0)) ** (1/3)

    fig, axs = _plt.subplots(4, 1, figsize=(10, 10), sharex=False)
    axs[0].plot(rtts_arr, label="RTT (ms)")
    axs[0].set_title(f"RSM-Ping — {host}")
    axs[0].set_ylabel("ms"); axs[0].legend()

    axs[1].plot(jitter_var_roll, label="Rolling jitter var (ms^2)"); axs[1].axhline(thr, ls="--", label="Threshold (eff)")
    axs[1].set_ylabel("ms^2"); axs[1].legend()

    axs[2].plot(H_roll, label="Spectral entropy (0=peaked)")
    axs[2].plot(D_roll, label="Autocorr decay (lower=coherent)")
    axs[2].set_ylabel("norm"); axs[2].legend()

    axs[3].plot(R_roll, label="Resonant score (R)")
    axs[3].set_xlabel("Sample (~ping #)"); axs[3].set_ylabel("0–1"); axs[3].legend()

    fig.suptitle(f"S={result['stillness']:.3f} | R={result.get('resonant_score', float('nan')):.3f} | H={result.get('spectral_entropy', float('nan')):.3f} | D={result.get('autocorr_decay', float('nan')):.3f}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = os.path.join(outdir, f"rsm_ping_{host.replace('.', '_')}.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    _plt.close(fig)
    return outpath

def main():
    args = parse_args()
    # Acquire
    if args.mode in ("auto","ping"):
        rtts = run_ping_batch(args.host, args.count, args.interval)
        if args.mode == "auto" and len(rtts) == 0:
            rtts = synth_series(args.count, mode=args.simulate, base=20.0)
    else:
        rtts = synth_series(args.count, mode=args.simulate, base=20.0)

    if args.verbose:
        for i, r in enumerate(rtts):
            print(f"[{i:03d}] {r:.3f} ms")

    result = compute_resonant_stillness(rtts, args.threshold, adaptive=bool(args.adaptive), factor=args.factor)

    # Output summary
    print("="*70)
    print(f"Average Latency:   {result['avg_latency_ms']:.3f} ms")
    print(f"Jitter Variance:   {result['jitter_var_ms2']:.6f} ms^2")
    print(f"Stillness Score S: {result['stillness']:.3f}")
    print(f"Resonant Score  R: {result['resonant_score']:.3f}")
    print(f"Spectral Entropy : {result['spectral_entropy']:.3f} (0=peaked,1=flat)")
    print(f"Autocorr Decay  : {result['autocorr_decay']:.3f} (lower=coherent)")
    print(f"Threshold (eff) : {result.get('adaptive_threshold', args.threshold):.6f} ms^2")
    print(f"Samples:         {result['samples']}")
    print("="*70)

    # Save files
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = save_csv(args.outdir, args.csv, args.host, rtts, result, args.threshold)
    print(f"CSV saved:  {csv_path}")
    if int(args.plot) == 1:
        plot_path = save_plot(args.outdir, args.host, rtts, result)
        if plot_path: print(f"Plot saved: {plot_path}")

if __name__ == "__main__":
    main()
