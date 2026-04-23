#!/usr/bin/env python3
"""
General Packet Sniffer with ML Detection
Captures all network packets, aggregates them into flows, extracts specified features,
and uses a pre-trained Random Forest model and a feature selector for attack prediction.
"""

import sys
import pickle
import csv
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
import re
from scapy.all import sniff, Raw
from scapy.layers.dns import DNS, DNSQR
from scapy.layers.inet import IP, UDP, TCP
from datetime import datetime

warnings.filterwarnings('ignore')

class FlowAnalyzer:
    def __init__(self, model_path="random_forest_model.pkl", selector_path="feature_selector.pkl", encoder_path="label_encoder.pkl"):
        self.model = self.load_pickle(model_path)
        self.selector = self.load_pickle(selector_path)
        self.label_encoder = self.load_pickle(encoder_path)
        self.csv_file = "flow_predictions.csv"
        
        self.feature_names = [
            'Destination Port', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
            'Bwd Packet Length Min', 'Bwd Packet Length Std', 'Flow Bytes/s',
            'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Min', 'Fwd IAT Total',
            'Fwd IAT Std', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Std',
            'Bwd IAT Max', 'Bwd IAT Min', 'Bwd Header Length', 'Fwd Packets/s',
            'Bwd Packets/s', 'Min Packet Length', 'FIN Flag Count',
            'SYN Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
            'Avg Fwd Segment Size', 'Fwd Header Length.1', 'Subflow Fwd Packets',
            'Subflow Fwd Bytes', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
            'Active Std', 'Active Max', 'Active Min', 'Idle Std', 'Label', 'count',
            'Flow Bytes/s_had_inf', 'Flow Bytes/s_was_missing',
            'Flow Packets/s_had_inf'
        ]
        
        # We don't train 'Label' so we remove it from inference features
        self.inference_features = [f for f in self.feature_names if f != 'Label']
        self.init_csv()

    def load_pickle(self, path):
        try:
            import joblib
            return joblib.load(path)
        except Exception as e:
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e2:
                print(f"Error loading {path}: joblib error: {e}, pickle error: {e2}")
                return None

    def init_csv(self):
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Source IP', 'Destination IP', 'Domain Name', 'Protocol', 'Prediction', 'Confidence'])

    def predict_flow(self, flow_key, flow_features, domain_name="N/A"):
        if not self.model or not self.selector:
            return None, None
        
        try:
            # Create a DataFrame for exactly what the selector expects
            df = pd.DataFrame([flow_features], columns=self.inference_features)
            
            # Apply feature selection
            selected_features = self.selector.transform(df)
            
            # Predict
            pred = self.model.predict(selected_features)[0]
            
            # Get confidence if predict_proba is available
            confidence = 0.0
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(selected_features)[0]
                confidence = max(probs)
                
            # Decode label if label encoder is loaded
            if self.label_encoder and hasattr(self.label_encoder, "inverse_transform"):
                try:
                    pred_label = self.label_encoder.inverse_transform([pred])[0]
                except Exception:
                    pred_label = pred
            else:
                pred_label = pred
                
            # Format protocol
            proto_num = flow_key[4]
            if proto_num == 6:
                proto_str = "TCP"
            elif proto_num == 17:
                proto_str = "UDP"
            else:
                proto_str = str(proto_num)
            
            # Log to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    f"{flow_key[0]}:{flow_key[2]}",
                    f"{flow_key[1]}:{flow_key[3]}",
                    domain_name,
                    proto_str,
                    pred_label,
                    confidence
                ])
                
            return pred_label, confidence
        except Exception as e:
            print(f"Prediction error for flow {flow_key}: {e}")
            return None, None

class PacketSniffer:
    def __init__(self, interface=None, timeout=60, count=0):
        self.interface = interface
        self.timeout = timeout
        self.count = count
        self.analyzer = FlowAnalyzer()
        
        # Track flows: 5-tuple -> data
        # Data will store packet metadata to calculate features when flow ends or timeouts
        self.flows = {}
        
    def get_5_tuple(self, packet):
        if not packet.haslayer(IP):
            return None
            
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
        
        src_port, dst_port = 0, 0
        if packet.haslayer(TCP):
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif packet.haslayer(UDP):
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            return None
            
        return (src_ip, dst_ip, src_port, dst_port, proto)

    def extract_flags(self, packet):
        flags = {'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0, 'ACK': 0, 'URG': 0, 'ECE': 0, 'CWR': 0}
        if packet.haslayer(TCP):
            f = packet[TCP].flags
            if f & 0x01: flags['FIN'] = 1
            if f & 0x02: flags['SYN'] = 1
            if f & 0x04: flags['RST'] = 1
            if f & 0x08: flags['PSH'] = 1
            if f & 0x10: flags['ACK'] = 1
            if f & 0x20: flags['URG'] = 1
            if f & 0x40: flags['ECE'] = 1
            if f & 0x80: flags['CWR'] = 1
        return flags

    def process_packet(self, packet):
        flow_key = self.get_5_tuple(packet)
        if not flow_key:
            return
            
        # For simplicity, we define a flow direction by the first packet seen
        # If we see the reverse tuple, we treat it as the same flow but backward
        reverse_key = (flow_key[1], flow_key[0], flow_key[3], flow_key[2], flow_key[4])
        
        is_fwd = True
        active_key = flow_key
        if reverse_key in self.flows:
            is_fwd = False
            active_key = reverse_key
        elif flow_key not in self.flows:
            # Initialize new flow
            self.flows[flow_key] = {
                'start_time': packet.time,
                'last_time': packet.time,
                'fwd_pkts': [],
                'bwd_pkts': [],
                'fwd_iat': [],
                'bwd_iat': [],
                'flow_iat': [],
                'flags': defaultdict(int),
                'fwd_hdr_len': 0,
                'bwd_hdr_len': 0,
                'init_win_fwd': packet[TCP].window if packet.haslayer(TCP) else 0,
                'init_win_bwd': -1,
                'act_data_pkt_fwd': 0,
                'min_seg_size_fwd': len(packet[TCP]) if packet.haslayer(TCP) else 0,
                'domain': "N/A"
            }
        
        flow = self.flows[active_key]
        
        # Try extracting DNS domain name if applicable
        if packet.haslayer('DNSQR'):
            try:
                domain = packet['DNSQR'].qname.decode('utf-8').rstrip('.')
                if domain:
                    flow['domain'] = domain
            except Exception:
                pass                
        # Try extracting SNI from TLS (HTTPS) or HTTP Host header
        if packet.haslayer(Raw):
            payload = packet[Raw].load
            # Check for HTTP Host header
            if flow_key[3] == 80 or flow_key[2] == 80:
                try:
                    host = re.search(b'Host: (.*?)\r\n', payload)
                    if host:
                        flow['domain'] = host.group(1).decode('utf-8')
                except Exception:
                    pass
            # Check for TLS SNI
            elif flow_key[3] == 443 or flow_key[2] == 443:
                try:
                    # Very basic pattern for TLS SNI extraction over Raw payload
                    # This relies on standard TLS 1.2+ Client Hello structure
                    if len(payload) > 43 and payload[0] == 0x16 and payload[5] == 0x01:
                        # Standard SNI extension format: 
                        # \x00\x00 (Ext Type 0) + 2 bytes (ext len) + 2 bytes (list len) + \x00 (name type) + 2 bytes (name len) + domain
                        sni_match = re.search(b'\x00\x00....\x00..([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})', payload, re.DOTALL)
                        if sni_match:
                            domain = sni_match.group(1).decode('utf-8')
                            if domain:
                                flow['domain'] = domain
                except Exception:
                    pass        
        # Calculate IAT
        current_time = float(packet.time)
        flow['flow_iat'].append(current_time - float(flow['last_time']))
        
        pkt_len = len(packet)
        hdr_len = len(packet[IP]) - len(packet[IP].payload)
        
        # Flags
        flags = self.extract_flags(packet)
        for k, v in flags.items():
            flow['flags'][k] += v

        if is_fwd:
            if len(flow['fwd_pkts']) > 0:
                flow['fwd_iat'].append(current_time - float(flow['last_time']))
            flow['fwd_pkts'].append(pkt_len)
            flow['fwd_hdr_len'] += hdr_len
            if pkt_len > 0:
                flow['act_data_pkt_fwd'] += 1
        else:
            if len(flow['bwd_pkts']) > 0:
                flow['bwd_iat'].append(current_time - float(flow['last_time']))
            flow['bwd_pkts'].append(pkt_len)
            flow['bwd_hdr_len'] += hdr_len
            if flow['init_win_bwd'] == -1 and packet.haslayer(TCP):
                flow['init_win_bwd'] = packet[TCP].window
                
        flow['last_time'] = current_time
        
        # Optional: check if flow has ended (FIN or RST) or time threshold
        if flags['FIN'] > 0 or flags['RST'] > 0:
            self.finalize_flow(active_key)

    def _safe_calc(self, data, func, default=0.0):
        if not data: return default
        if func == 'max': return float(np.max(data))
        if func == 'min': return float(np.min(data))
        if func == 'mean': return float(np.mean(data))
        if func == 'std': return float(np.std(data))
        if func == 'total': return float(np.sum(data))
        return default

    def finalize_flow(self, flow_key):
        flow = self.flows.pop(flow_key, None)
        if not flow:
            return
            
        duration = max(float(flow['last_time']) - float(flow['start_time']), 0.000001)
        tot_fwd_bytes = sum(flow['fwd_pkts'])
        tot_bwd_bytes = sum(flow['bwd_pkts'])
        tot_bytes = tot_fwd_bytes + tot_bwd_bytes
        tot_pkts = len(flow['fwd_pkts']) + len(flow['bwd_pkts'])
        
        all_pkts = flow['fwd_pkts'] + flow['bwd_pkts']
        
        features = {}
        features['Destination Port'] = flow_key[3]
        features['Fwd Packet Length Max'] = self._safe_calc(flow['fwd_pkts'], 'max')
        features['Fwd Packet Length Min'] = self._safe_calc(flow['fwd_pkts'], 'min')
        features['Bwd Packet Length Min'] = self._safe_calc(flow['bwd_pkts'], 'min')
        features['Bwd Packet Length Std'] = self._safe_calc(flow['bwd_pkts'], 'std')
        
        features['Flow Bytes/s'] = tot_bytes / duration
        features['Flow Packets/s'] = tot_pkts / duration
        features['Flow IAT Mean'] = self._safe_calc(flow['flow_iat'], 'mean')
        features['Flow IAT Min'] = self._safe_calc(flow['flow_iat'], 'min')
        
        features['Fwd IAT Total'] = self._safe_calc(flow['fwd_iat'], 'total')
        features['Fwd IAT Std'] = self._safe_calc(flow['fwd_iat'], 'std')
        features['Fwd IAT Min'] = self._safe_calc(flow['fwd_iat'], 'min')
        
        features['Bwd IAT Total'] = self._safe_calc(flow['bwd_iat'], 'total')
        features['Bwd IAT Std'] = self._safe_calc(flow['bwd_iat'], 'std')
        features['Bwd IAT Max'] = self._safe_calc(flow['bwd_iat'], 'max')
        features['Bwd IAT Min'] = self._safe_calc(flow['bwd_iat'], 'min')
        
        features['Bwd Header Length'] = flow['bwd_hdr_len']
        features['Fwd Packets/s'] = len(flow['fwd_pkts']) / duration
        features['Bwd Packets/s'] = len(flow['bwd_pkts']) / duration
        features['Min Packet Length'] = self._safe_calc(all_pkts, 'min')
        
        features['FIN Flag Count'] = flow['flags']['FIN']
        features['SYN Flag Count'] = flow['flags']['SYN']
        features['PSH Flag Count'] = flow['flags']['PSH']
        features['ACK Flag Count'] = flow['flags']['ACK']
        features['URG Flag Count'] = flow['flags']['URG']
        features['CWE Flag Count'] = flow['flags']['CWR']
        features['ECE Flag Count'] = flow['flags']['ECE']
        
        features['Down/Up Ratio'] = len(flow['bwd_pkts']) / max(len(flow['fwd_pkts']), 1)
        features['Avg Fwd Segment Size'] = self._safe_calc(flow['fwd_pkts'], 'mean')
        features['Fwd Header Length.1'] = flow['fwd_hdr_len'] # Same as Fwd Header Length usually
        
        features['Subflow Fwd Packets'] = len(flow['fwd_pkts'])
        features['Subflow Fwd Bytes'] = tot_fwd_bytes
        features['Init_Win_bytes_forward'] = flow['init_win_fwd']
        features['Init_Win_bytes_backward'] = flow['init_win_bwd']
        features['act_data_pkt_fwd'] = flow['act_data_pkt_fwd']
        features['min_seg_size_forward'] = flow['min_seg_size_fwd']
        
        features['Active Std'] = 0.0 # Would need sub-flow active/idle tracking
        features['Active Max'] = 0.0
        features['Active Min'] = 0.0
        features['Idle Std'] = 0.0
        features['count'] = 1
        
        features['Flow Bytes/s_had_inf'] = 1 if np.isinf(features['Flow Bytes/s']) else 0
        features['Flow Bytes/s_was_missing'] = 1 if np.isnan(features['Flow Bytes/s']) else 0
        features['Flow Packets/s_had_inf'] = 1 if np.isinf(features['Flow Packets/s']) else 0
        
        # Replace inf with 0 or a large number for prediction
        for k, v in features.items():
            if np.isinf(v) or np.isnan(v):
                features[k] = 0.0

        # Vector in expected order
        predict_vector = [features[k] for k in self.analyzer.inference_features]
        
        # Run prediction
        domain = flow.get('domain', "N/A")
        pred, confidence = self.analyzer.predict_flow(flow_key, predict_vector, domain)
        
        # Format protocol
        proto_num = flow_key[4]
        if proto_num == 6:
            proto_str = "TCP"
        elif proto_num == 17:
            proto_str = "UDP"
        else:
            proto_str = str(proto_num)
            
        src = f"{flow_key[0]}:{flow_key[2]}"
        dst = f"{flow_key[1]}:{flow_key[3]}"
        
        conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        pred_str = str(pred) if pred is not None else "None"
        
        print(f"{src:<25} {dst:<25} {proto_str:<7} {pred_str:<15} {conf_str}")

    def start(self):
        print("Starting general packet sniffer...")
        print(f"{'Source':<25} {'Destination':<25} {'Proto':<7} {'Prediction':<15} {'Confidence'}")
        print("-" * 85)
        try:
            sniff(
                prn=self.process_packet,
                store=False,
                count=self.count,
                iface=self.interface
            )
        except KeyboardInterrupt:
            print("\nSniffing stopped by user. Finalizing all active flows...")
        finally:
            active_flows = list(self.flows.keys())
            for flow_key in active_flows:
                self.finalize_flow(flow_key)
            print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General Packet Sniffer for ML Detection")
    parser.add_argument("-i", "--interface", help="Network interface to sniff on")
    parser.add_argument("-c", "--count", type=int, default=0, help="Number of packets to capture (0 = infinite)")
    args = parser.parse_args()
    
    sniffer = PacketSniffer(interface=args.interface, count=args.count)
    sniffer.start()
