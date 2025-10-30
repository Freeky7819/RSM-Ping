# RSM-Ping (Open Edition)
**Resonant Stillness Monitor — Measuring network “calmness” through latency jitter.**  
[![PyPI](https://img.shields.io/pypi/v/rsm_ping.svg)](https://pypi.org/project/rsm_ping/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)](#)

---

## 🌊 Overview
**RSM-Ping** transforms raw ping jitter into four *resonant metrics* that describe the stability and coherence of a network or signal channel.

| Metric | Meaning | Interpretation |
|:--|:--|:--|
| **S** — *Stillness* | Variance-based stability | ↑ Higher = calmer |
| **H** — *Spectral Entropy* | Frequency dispersion | ↓ Lower = more focused |
| **D** — *Autocorrelation Decay* | Temporal coherence | ↓ Lower = more persistent |
| **R** — *Resonant Score* | Harmonic composite of S · (1–H) · (1–D) | ↑ Higher = resonant state |

When the system reaches a *resonant stillness*, latency variations become coherent rather than random — energy focuses into stable oscillations instead of chaos.

---

## ⚙️ Installation
```bash
pip install rsm_ping
or from a local clone:

bash
Copy code
pip install -r requirements.txt
python rsm_ping.py --host 8.8.8.8 --count 150 --threshold 0.01 --plot 1
Dependencies:

nginx
Copy code
numpy
matplotlib
🚀 Quick Examples
Real network measurement
bash
Copy code
rsm-ping --host 8.8.8.8 --count 150 --threshold 0.01 --adaptive 1 --plot 1
Synthetic resonance mode
bash
Copy code
rsm-ping --mode synthetic --simulate resonant --count 200 --threshold 0.01 --plot 1
Outputs:

results/*.csv — latency + metric summary

results/*.png — 4-panel chart: RTT, jitter variance, H/D evolution, R score

## 🧠 Method (in short)

1. **Compute jitter variance and stillness**

   S = exp( -σ² / θ )

2. **Estimate spectral entropy**

   H = normalized entropy of FFT(ΔRTT)

3. **Find autocorrelation decay**

   D = time (normalized) when autocorrelation drops to 10 % of peak

4. **Combine harmonically**

   R = ( S · (1 − H) · (1 − D) )^(1/3)
 

When R → 1, the system operates near perfect coherence —
When R → 0, it’s dominated by random noise.

🧩 Example (PowerShell)
powershell
Copy code
# Real measurement with plot + CSV
rsm-ping --host 1.1.1.1 --count 120 --threshold 0.01 --plot 1

# Adaptive threshold (25th percentile × factor)
rsm-ping --host 8.8.4.4 --count 200 --adaptive 1 --factor 1.5 --plot 1

# Synthetic "resonant" test
rsm-ping --mode synthetic --simulate resonant --count 200 --plot 1
📄 License
Apache-2.0 © Freedom (Damjan Žakelj), 2025
