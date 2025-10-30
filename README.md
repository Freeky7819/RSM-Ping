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

pip install -r requirements.txt
python rsm_ping.py --host 8.8.8.8 --count 150 --threshold 0.01 --plot 1
```
Dependencies

numpy

matplotlib

🚀 Quick Examples
Real network measurement

```bash
rsm-ping --host 8.8.8.8 --count 150 --threshold 0.01 --adaptive 1 --plot 1
Synthetic resonance mode

rsm-ping --mode synthetic --simulate resonant --count 200 --threshold 0.01 --plot 1
Outputs

results/*.csv — latency + metric summary

results/*.png — 4-panel chart: RTT, jitter variance, H/D evolution, R score
```

🧠 Method (in short)
Compute jitter variance and stillness
S = exp( -σ² / θ )

Estimate spectral entropy
H = normalized entropy of FFT(ΔRTT)

Find autocorrelation decay
D = normalized time when autocorrelation drops to 10% of peak

Combine harmonically
R = ( S · (1 − H) · (1 − D) )^(1/3)


⚡️ PRO / SDK Edition
The RSM-Ping PRO Edition extends Open with:

Real-time live dashboard (Gradio UI)

Structured JSON export for ROC / ISM-X

SDK hooks for adaptive agents & network analytics

Optional Harmonic Logs (audit & provenance)


For collaboration or licensing inquiries:
📧 zakelj.damjan@gmail.com

Open Edition = free scientific tool.
PRO Edition = research-grade SDK with live resonance analytics.
