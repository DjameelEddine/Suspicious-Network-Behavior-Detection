# Suspicious Network Behavior Detection

A machine learning system trained on network traffic data to detect abnormal and potentially malicious behavior. The system analyzes per-flow traffic features — ports, protocols, packet sizes, duration, and more — and classifies each flow as either normal or suspicious, with attack type identification where possible.

---

## Problem

Network intrusion detection systems (NIDS) traditionally rely on rule-based signatures. They catch known attacks but miss novel ones. A learned model that understands what *normal* traffic looks like can flag anomalies without needing to know the attack in advance.

---

## Approach

```
Raw network traffic (PCAP / NetFlow)
        │
        ▼
Feature extraction per flow:
  ports, protocol, packet size (min/max/mean),
  duration, byte count, flag counts, IAT stats
        │
        ▼
ML classifier
        │
        ▼
Label: NORMAL / SUSPICIOUS
       (+ attack type if applicable)
```

---

## Features

| Category | Features |
|---|---|
| **Connection** | src/dst port, protocol, service type |
| **Volume** | total bytes sent/received, packet count |
| **Timing** | duration, inter-arrival time (mean, std) |
| **Packet-level** | min/max/mean packet size, flag distribution |
| **Ratios** | bytes per packet, packets per second |

---

## Models compared

- Random Forest
- Gradient Boosting (XGBoost)
- Logistic Regression (baseline)
- Isolation Forest (unsupervised baseline)

---

## Dataset

Trained and evaluated on public network intrusion datasets including **CICIDS** and **NSL-KDD**, which contain labeled normal and attack traffic across multiple attack categories (DoS, DDoS, port scan, brute force, infiltration).

### Data prep log

- Merged CICIDS day/time CSV files in chronological order into a single file using:

```bash
python merge_to_traffic.py
```

- Output file generated locally:
        - `traffic.csv`

- Prevented pushing large merged data to GitHub by adding this to `.gitignore`:

```gitignore
traffic.csv
```

- If `traffic.csv` is ever tracked in Git, untrack it without deleting the local file:

```bash
git rm --cached traffic.csv
```

---

## Setup

```bash
git clone https://github.com/nacermissouni23/network-behavior-detection
cd network-behavior-detection
pip install -r requirements.txt
```

```bash
# Train on labeled traffic dataset
python train.py --data data/cicids2017.csv

# Classify a new traffic file
python classify.py --input data/test_flows.csv --output predictions.csv

# Evaluate with full metrics
python evaluate.py --model models/rf_model.pkl --test data/test.csv
```

---

## Evaluation metrics

- Classification accuracy, F1-score per class
- Confusion matrix across attack types
- ROC-AUC
- False positive rate (critical for operational use)

---

## Stack

```
scikit-learn / XGBoost    classifiers, evaluation
scapy / pyshark           PCAP feature extraction
pandas / numpy            data preparation
matplotlib / seaborn      visualization
```

---

## Related project

See also: [DNS Tunneling Detection](https://github.com/nacermissouni23/dns-tunneling-detection) — specialized detection for covert DNS-based exfiltration.
