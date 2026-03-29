from __future__ import annotations

from collections import deque
from statistics import mean
from time import perf_counter
from typing import Deque, Dict

from .utils import health_status


class MetricsTracker:
    def __init__(self, latency_window: int = 1000) -> None:
        self.latency_window = latency_window
        self.start_time = perf_counter()
        self.latencies_ms: Deque[float] = deque(maxlen=latency_window)
        self.total_processed = 0
        self.benign_count = 0
        self.attack_count = 0
        self.invalid_count = 0
        self.error_count = 0
        self.runtime_status = "stopped"

    def reset(self) -> None:
        self.start_time = perf_counter()
        self.latencies_ms.clear()
        self.total_processed = 0
        self.benign_count = 0
        self.attack_count = 0
        self.invalid_count = 0
        self.error_count = 0
        self.runtime_status = "stopped"

    def set_runtime_status(self, status: str) -> None:
        self.runtime_status = status

    def record_prediction(self, latency_ms: float, is_attack: bool) -> None:
        self.total_processed += 1
        self.latencies_ms.append(float(latency_ms))
        if is_attack:
            self.attack_count += 1
        else:
            self.benign_count += 1

    def record_invalid(self, count: int = 1) -> None:
        self.invalid_count += max(0, count)

    def record_error(self, count: int = 1) -> None:
        self.error_count += max(0, count)

    def snapshot(self) -> Dict[str, float]:
        elapsed = max(perf_counter() - self.start_time, 1e-9)
        avg_latency = mean(self.latencies_ms) if self.latencies_ms else 0.0
        throughput = self.total_processed / elapsed

        denominator = max(self.total_processed + self.invalid_count, 1)
        error_rate = (self.error_count + self.invalid_count) / denominator

        return {
            "runtime_status": self.runtime_status,
            "elapsed_seconds": round(elapsed, 2),
            "processed_flows": self.total_processed,
            "benign_count": self.benign_count,
            "attack_count": self.attack_count,
            "invalid_count": self.invalid_count,
            "error_count": self.error_count,
            "avg_latency_ms": round(avg_latency, 2),
            "throughput_fps": round(throughput, 2),
            "error_rate": round(error_rate, 4),
            "health": health_status(avg_latency, error_rate),
        }
