# Suspicious-Network-Behavior-Detection
A machine learning system trained on network traffic data to detect abnormal or malicious behavior. The system would analyze traffic features (such as ports, protocols, packet size, and duration) and classify flows as normal or suspicious, with possible attack types.

# Steps To try the tool
## Save the model

- Run the notebok `snba_cns2.ipynb`

## Run the packets sniffer

Use the command:
 ```bash
python generale_packet_sniffer.py
```

## Run the streamlit app

1. Install streamlit first
Use the command:
 ```bash
pip install streamlit
```

2. Run the streamlit web app
Use the command:
 ```bash
`streamlit app.py`
 ```
