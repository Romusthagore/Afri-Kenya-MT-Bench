import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


class TrainingMonitor:
    def __init__(self):
        self.start_time = None
        self.gpu_name = None
        self.gpu_memory_gb = None

    def start(self):
        self.start_time = time.perf_counter()

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            self.gpu_name = torch.cuda.get_device_name(device)
            self.gpu_memory_gb = round(
                torch.cuda.get_device_properties(device).total_memory / 1024**3,
                2,
            )
            torch.cuda.reset_peak_memory_stats(device)

    def stop(self, output_dir, language, steps):
        if self.start_time is None:
            raise RuntimeError("Monitor was not started")

        duration = time.perf_counter() - self.start_time

        report = {
            "language": language,
            "steps": steps,
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2),
            "duration_hours": round(duration / 3600, 4),
            "steps_per_second": round(steps / duration, 4),
            "gpu_name": self.gpu_name,
            "gpu_total_memory_gb": self.gpu_memory_gb,
            "gpu_peak_allocated_gb": None,
            "gpu_peak_reserved_gb": None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        if torch.cuda.is_available():
            report["gpu_peak_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            )
            report["gpu_peak_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / 1024**3, 2
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "training_resources.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\nTRAINING RESOURCE REPORT")
        for key, value in report.items():
            print(f"{key}: {value}")

        print("Saved:", path)

        return report
