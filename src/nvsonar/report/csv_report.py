"""CSV report output"""

import csv
import io

from nvsonar.monitor import Metrics
from nvsonar.monitor.hardware import GPUInfo
from nvsonar.analysis.bottleneck import BottleneckResult
from nvsonar.report.json import build_report

CSV_FIELDS = [
    "gpu_index",
    "gpu_name",
    "uuid",
    "driver_version",
    "cuda_version",
    "gpu_utilization",
    "memory_utilization",
    "memory_used_mb",
    "memory_total_mb",
    "memory_used_pct",
    "gpu_clock_mhz",
    "max_gpu_clock_mhz",
    "clock_reduction_pct",
    "temperature_c",
    "power_usage_w",
    "power_limit_w",
    "power_used_pct",
    "fan_speed_pct",
    "is_throttled",
    "throttle_severity",
    "pcie_gen",
    "pcie_width",
    "pcie_degraded",
    "ecc_enabled",
    "ecc_correctable",
    "ecc_uncorrectable",
    "bottleneck",
    "bottleneck_confidence",
    "bottleneck_detail",
]


def report_to_csv_row(
    gpu_info: GPUInfo,
    metrics: Metrics,
    bottleneck: BottleneckResult,
) -> dict:
    """Build a flat dict suitable for CSV output"""
    report = build_report(gpu_info, metrics, bottleneck)
    gpu = report["gpu"]
    m = report["metrics"]
    th = report["throttle"]
    pcie = report["pcie"]
    ecc = report["ecc"]
    analysis = report["analysis"]

    return {
        "gpu_index": gpu["index"],
        "gpu_name": gpu["name"],
        "uuid": gpu["uuid"],
        "driver_version": gpu["driver"],
        "cuda_version": gpu["cuda"],
        "gpu_utilization": m["gpu_utilization"],
        "memory_utilization": m["memory_utilization"],
        "memory_used_mb": m["memory_used_mb"],
        "memory_total_mb": m["memory_total_mb"],
        "memory_used_pct": m["memory_used_pct"],
        "gpu_clock_mhz": m["gpu_clock_mhz"],
        "max_gpu_clock_mhz": m["max_gpu_clock_mhz"],
        "clock_reduction_pct": m["clock_reduction_pct"],
        "temperature_c": m["temperature_c"],
        "power_usage_w": m["power_usage_w"],
        "power_limit_w": m["power_limit_w"],
        "power_used_pct": m["power_used_pct"],
        "fan_speed_pct": m["fan_speed_pct"],
        "is_throttled": th["is_throttled"],
        "throttle_severity": th["worst_severity"],
        "pcie_gen": f"{pcie['current_gen']}/{pcie['max_gen']}",
        "pcie_width": f"x{pcie['current_width']}/x{pcie['max_width']}",
        "pcie_degraded": pcie["is_degraded"],
        "ecc_enabled": ecc["enabled"],
        "ecc_correctable": ecc["correctable"],
        "ecc_uncorrectable": ecc["uncorrectable"],
        "bottleneck": analysis["bottleneck"],
        "bottleneck_confidence": analysis["confidence"],
        "bottleneck_detail": analysis["detail"],
    }


def to_csv(rows: list[dict]) -> str:
    """Serialize rows to CSV string with header"""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
