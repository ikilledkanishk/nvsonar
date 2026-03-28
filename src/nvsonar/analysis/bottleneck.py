"""GPU bottleneck classification from single metrics snapshot"""

from dataclasses import dataclass
from enum import Enum

from nvsonar.monitor import Metrics


class BottleneckType(Enum):
    IDLE = "idle"
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BANDWIDTH_BOUND = "memory_bandwidth_bound"
    MEMORY_CAPACITY_BOUND = "memory_capacity_bound"
    POWER_LIMITED = "power_limited"
    THERMAL_THROTTLED = "thermal_throttled"
    DATA_STARVED = "data_starved"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


@dataclass
class BottleneckResult:
    """Bottleneck classification with confidence and warnings"""

    bottleneck: BottleneckType
    confidence: float  # 0.0 - 1.0
    detail: str
    warnings: list[str]


def classify(metrics: Metrics) -> BottleneckResult:
    """Classify GPU bottleneck from a single metrics snapshot"""
    warnings = _collect_warnings(metrics)

    gpu_util = metrics.gpu_utilization
    mem_util = metrics.memory_utilization
    mem_used_pct = metrics.memory_used_pct
    power_pct = metrics.power_used_pct
    clock_drop = metrics.clock_reduction_pct
    throttle = metrics.throttle

    # --- idle ---
    if gpu_util < 5 and mem_util < 5:
        return BottleneckResult(
            BottleneckType.IDLE,
            0.95,
            "GPU has no active workload",
            warnings,
        )

    # --- hardware throttle (highest priority, masks real workload profile) ---

    # thermal throttle: HW or SW thermal bits
    hw_thermal = throttle.worst_severity == "critical" and any(
        r.name in ("Hardware Thermal Slowdown", "Hardware Slowdown")
        for r in throttle.active_reasons
    )
    sw_thermal = any(
        r.name == "Software Thermal Slowdown" for r in throttle.active_reasons
    )
    if hw_thermal:
        return BottleneckResult(
            BottleneckType.THERMAL_THROTTLED,
            0.95,
            f"Hardware thermal throttle active at {metrics.temperature}C, "
            f"clocks reduced {clock_drop:.0f}%",
            warnings,
        )
    if sw_thermal:
        return BottleneckResult(
            BottleneckType.THERMAL_THROTTLED,
            0.85,
            f"Driver thermal throttle at {metrics.temperature}C, "
            f"clocks reduced {clock_drop:.0f}%",
            warnings,
        )

    # power limited: SwPowerCap bit + power near limit
    sw_power_cap = any(r.name == "Software Power Cap" for r in throttle.active_reasons)
    if sw_power_cap and power_pct is not None and power_pct > 90:
        return BottleneckResult(
            BottleneckType.POWER_LIMITED,
            0.90,
            f"Power draw at {power_pct:.0f}% of limit, "
            f"clocks reduced {clock_drop:.0f}%",
            warnings,
        )

    # power limited without throttle bit but near TDP
    if power_pct is not None and power_pct > 95:
        return BottleneckResult(
            BottleneckType.POWER_LIMITED,
            0.80,
            f"Power at {power_pct:.0f}% of limit",
            warnings,
        )

    # --- workload classification ---

    # memory capacity bound: VRAM nearly full, OOM risk
    if mem_used_pct > 95:
        return BottleneckResult(
            BottleneckType.MEMORY_CAPACITY_BOUND,
            0.90,
            f"VRAM {mem_used_pct:.0f}% full, OOM risk",
            warnings,
        )

    # compute bound: GPU busy, memory controller not saturated
    if gpu_util > 85 and mem_util < 50:
        conf = 0.85 if gpu_util > 95 else 0.75
        return BottleneckResult(
            BottleneckType.COMPUTE_BOUND,
            conf,
            f"GPU {gpu_util}% utilized, memory controller only {mem_util}%",
            warnings,
        )

    # memory bandwidth bound: memory controller saturated, GPU waiting
    if mem_util > 80 and gpu_util < 85:
        conf = 0.85 if mem_util > 90 else 0.75
        return BottleneckResult(
            BottleneckType.MEMORY_BANDWIDTH_BOUND,
            conf,
            f"Memory controller {mem_util}% busy, GPU at {gpu_util}%",
            warnings,
        )

    # data starved: GPU underutilized but VRAM is allocated
    # classic sign of CPU/IO bottleneck
    if gpu_util < 40 and mem_used_pct > 50:
        return BottleneckResult(
            BottleneckType.DATA_STARVED,
            0.70,
            f"GPU only {gpu_util}% utilized despite {mem_used_pct:.0f}% VRAM used, "
            f"likely CPU or data loading bottleneck",
            warnings,
        )

    # balanced: both GPU and memory controller active
    if gpu_util > 70 and mem_util > 60:
        return BottleneckResult(
            BottleneckType.BALANCED,
            0.65,
            f"Balanced workload, GPU {gpu_util}%, memory controller {mem_util}%",
            warnings,
        )

    # low utilization but doing something
    if gpu_util >= 5 or mem_util >= 5:
        return BottleneckResult(
            BottleneckType.BALANCED,
            0.40,
            f"Light workload, GPU {gpu_util}%, memory controller {mem_util}%",
            warnings,
        )

    return BottleneckResult(
        BottleneckType.UNKNOWN,
        0.20,
        "Unable to classify workload from available metrics",
        warnings,
    )


def _collect_warnings(metrics: Metrics) -> list[str]:
    """Check for secondary issues alongside the main diagnosis"""
    warnings = []

    # misleading utilization: GPU shows busy but power is low
    if (
        metrics.gpu_utilization > 80
        and metrics.power_used_pct is not None
        and metrics.power_used_pct < 40
    ):
        warnings.append(
            f"GPU utilization is {metrics.gpu_utilization}% but power draw is only "
            f"{metrics.power_used_pct:.0f}%, GPU is not truly compute-saturated"
        )

    # PCIe degraded
    if metrics.pcie.is_degraded:
        warnings.append(metrics.pcie.degradation_reason)

    # ECC errors
    if metrics.ecc.uncorrectable > 0:
        warnings.append(
            f"{metrics.ecc.uncorrectable} uncorrectable ECC errors, "
            f"hardware may need replacement"
        )
    elif metrics.ecc.correctable > 0:
        warnings.append(
            f"{metrics.ecc.correctable} correctable ECC errors, monitor for increase"
        )

    # high clock reduction without throttle reason
    if metrics.clock_reduction_pct > 15 and not metrics.throttle.is_throttled:
        warnings.append(
            f"Clocks {metrics.clock_reduction_pct:.0f}% below max "
            f"without active throttle reason"
        )

    # fan speed
    if metrics.fan_speed is not None and metrics.fan_speed > 90:
        warnings.append(f"Fan at {metrics.fan_speed}%, thermal stress")

    return warnings
