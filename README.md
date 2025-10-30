
# RSM-Ping (Open Edition)
**Resonant Stillness Monitor — Open Source (Apache-2.0)**

RSM-Ping meri *mirnost* (stillness) omrežja/sistema iz RTT jitterja in izračuna 4 metrike:
- **S** (Stillness): odziv na jitter varianco (večje je bolje)
- **H** (Spectral entropy): razpršenost frekvenc (nižja je bolje)
- **D** (Autocorr decay): hitrost razpada korelacije (nižja je bolje)
- **R** (Resonant score): harmonična kompozicija S·(1−H)·(1−D)

## Namestitev (pip)
```bash
pip install rsm_ping
```

Alternativa (lokalno iz klona):
```bash
pip install -r requirements.txt
python rsm_ping.py --host 8.8.8.8 --count 150 --threshold 0.01 --plot 1
```

## Primeri
Realni ping z grafom:
```bash
rsm-ping --host 8.8.8.8 --count 150 --threshold 0.01 --adaptive 1 --plot 1
```

Sintetični test (resonant mode):
```bash
rsm-ping --mode synthetic --simulate resonant --count 150 --threshold 0.01 --plot 1
```

## Izhodi
- `results/*.csv` — RTT in povzetek metrik
- `results/*.png` — 4-panel graf (RTT, jitter var, entropy/decay, R)

## Licenca
Apache-2.0 © Freedom (Damjan), 2025

## Harmonic Signature Protocol
```json
{
  "omega": 6.0,
  "phi": 1.047,
  "gamma": 0.0,
  "intent": "resonant_network_stillness",
  "author": "Freedom (Damjan)"
}
```
