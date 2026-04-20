"""
IoT-Shield Feature Engineering Module
======================================
Derives additional features from the raw 37 CICIOT2023 features.

CRITICAL: This module MUST be used identically in:
  - data_loader.py (during training)
  - detector.py   (during inference)

Changing this file requires retraining the model.
"""

import numpy as np
import pandas as pd
from typing import List

# ═══════════════════════════════════════════════════════════
# Feature engineering version — bump when formulas change
# ═══════════════════════════════════════════════════════════
FEATURE_ENG_VERSION = "1.0.0"

# Small epsilon to avoid division by zero
EPS = 1e-6


# List of derived feature names, in the exact order they'll be appended.
# This is THE canonical order — detector.py must use this same order.
DERIVED_FEATURES: List[str] = [
    "syn_to_fin_ratio",        # SYN flood signature
    "ack_to_syn_ratio",        # Normal handshake balance
    "rst_ratio",               # Connection reset rate
    "size_variance_ratio",     # Packet size irregularity
    "size_range",              # Max - Min packet size
    "pkts_per_second",         # Rate intensity
    "bytes_per_packet",        # Average payload ratio
    "flag_diversity",          # How many flag types seen
    "has_web_traffic",         # HTTP or HTTPS present
    "has_system_traffic",      # SSH, Telnet, DNS present
    "protocol_mix",            # Number of distinct protocols
    "high_frequency_flag",     # Rate > threshold indicator
]


def add_derived_features_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas DataFrame version — used in data_loader.py during training.
    Accepts a DataFrame with the 37 raw CICIOT2023 columns.
    Returns a DataFrame with 37 + len(DERIVED_FEATURES) columns.
    """
    X = X.copy()

    # Rate-based ratios (flag activity)
    X["syn_to_fin_ratio"] = X["syn_count"] / (X["fin_count"] + EPS)
    X["ack_to_syn_ratio"] = X["ack_count"] / (X["syn_count"] + EPS)
    X["rst_ratio"] = X["rst_count"] / (X["Number"] + EPS)

    # Packet size features
    X["size_variance_ratio"] = X["Variance"] / (X["AVG"] + EPS)
    X["size_range"] = X["Max"] - X["Min"]
    X["bytes_per_packet"] = X["Tot size"] / (X["Number"] + EPS)

    # Flow intensity
    X["pkts_per_second"] = X["Rate"]  # Rate is already pkts/sec in CICIOT2023

    # Protocol mix
    protocol_cols = ["HTTP", "HTTPS", "DNS", "SSH", "Telnet", "SMTP", "IRC",
                     "TCP", "UDP", "DHCP", "ARP", "ICMP"]
    available_protos = [c for c in protocol_cols if c in X.columns]
    if available_protos:
        X["protocol_mix"] = X[available_protos].sum(axis=1)
    else:
        X["protocol_mix"] = 0.0

    # Flag diversity (how many distinct flag types are set)
    flag_cols = ["fin_flag_number", "syn_flag_number", "rst_flag_number",
                 "psh_flag_number", "ack_flag_number", "ece_flag_number",
                 "cwr_flag_number"]
    available_flags = [c for c in flag_cols if c in X.columns]
    if available_flags:
        X["flag_diversity"] = X[available_flags].sum(axis=1)
    else:
        X["flag_diversity"] = 0.0

    # Web / system traffic flags
    X["has_web_traffic"] = ((X.get("HTTP", 0) > 0) | (X.get("HTTPS", 0) > 0)).astype(np.float32)
    X["has_system_traffic"] = (
        (X.get("SSH", 0) > 0)
        | (X.get("Telnet", 0) > 0)
        | (X.get("DNS", 0) > 0)
    ).astype(np.float32)

    # High-frequency indicator (Rate > 1000 pkts/sec → likely flood)
    X["high_frequency_flag"] = (X["Rate"] > 1000.0).astype(np.float32)

    # Clean up infinities / NaNs produced by ratios
    for col in DERIVED_FEATURES:
        if col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            X[col] = X[col].astype(np.float32)

    return X


def add_derived_features_np(features: np.ndarray, feature_order: List[str]) -> np.ndarray:
    """
    NumPy version — used in detector.py during inference.
    Accepts a 1D array of 37 raw feature values (in feature_order).
    Returns a 1D array of 37 + len(DERIVED_FEATURES) values.

    feature_order: the order of the raw features in `features` array.
    """
    # Build lookup dict
    fmap = {name: float(features[i]) for i, name in enumerate(feature_order)}

    def g(name: str, default: float = 0.0) -> float:
        return fmap.get(name, default)

    derived = [
        g("syn_count") / (g("fin_count") + EPS),                              # syn_to_fin_ratio
        g("ack_count") / (g("syn_count") + EPS),                              # ack_to_syn_ratio
        g("rst_count") / (g("Number") + EPS),                                 # rst_ratio
        g("Variance") / (g("AVG") + EPS),                                     # size_variance_ratio
        g("Max") - g("Min"),                                                  # size_range
        g("Rate"),                                                            # pkts_per_second
        g("Tot size") / (g("Number") + EPS),                                  # bytes_per_packet
        (g("fin_flag_number") + g("syn_flag_number") + g("rst_flag_number")
         + g("psh_flag_number") + g("ack_flag_number")
         + g("ece_flag_number") + g("cwr_flag_number")),                      # flag_diversity
        1.0 if (g("HTTP") > 0 or g("HTTPS") > 0) else 0.0,                    # has_web_traffic
        1.0 if (g("SSH") > 0 or g("Telnet") > 0 or g("DNS") > 0) else 0.0,    # has_system_traffic
        (g("HTTP") + g("HTTPS") + g("DNS") + g("SSH") + g("Telnet")
         + g("SMTP") + g("IRC") + g("TCP") + g("UDP") + g("DHCP")
         + g("ARP") + g("ICMP")),                                             # protocol_mix
        1.0 if g("Rate") > 1000.0 else 0.0,                                   # high_frequency_flag
    ]

    # Clean infinities / NaNs
    derived_clean = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in derived]

    return np.concatenate([features, np.array(derived_clean, dtype=np.float32)])
