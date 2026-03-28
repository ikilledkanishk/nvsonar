"""Main TUI application"""

from collections import deque
from dataclasses import dataclass
from time import time

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from nvsonar.monitor import (
    initialize,
    get_device_count,
    get_gpu_info,
    list_gpus,
    MetricsCollector,
)
from nvsonar.analysis import classify, TemporalAnalyzer, recommend

UPDATE_INTERVAL = 0.5
PEAK_WINDOW = 60.0


def _make_bar(value: float, max_value: float, width: int = 20) -> str:
    """Create a text progress bar"""
    if max_value <= 0:
        ratio = 0.0
    else:
        ratio = min(value / max_value, 1.0)

    filled = int(ratio * width)
    empty = width - filled
    return "█" * filled + "░" * empty


@dataclass
class MetricSnapshot:
    """Single metric snapshot with timestamp"""

    timestamp: float
    temperature: float
    power_usage: float
    gpu_utilization: int
    memory_utilization: int
    memory_used: int
    gpu_clock: int
    memory_clock: int
    bottleneck: str | None = None


class DeviceList(Static):
    """Display available GPUs"""

    def on_mount(self) -> None:
        if not initialize():
            self.update("[red]Failed to initialize NVML[/red]")
            return

        devices = list_gpus()
        if not devices:
            self.update("[yellow]No GPUs found[/yellow]")
            return

        table = Table(title="Available GPUs")
        table.add_column("Index", style="cyan", justify="center")
        table.add_column("Name", style="green")
        table.add_column("Memory", style="yellow", justify="right")
        table.add_column("Driver", style="magenta")
        table.add_column("CUDA", style="blue")

        for device in devices:
            memory_gb = device.memory_total / (1024**3)
            table.add_row(
                str(device.index),
                device.name,
                f"{memory_gb:.1f} GB",
                device.driver_version,
                device.cuda_version,
            )

        self.update(table)


class LiveMetrics(Static):
    """Display live metrics for all GPUs"""

    def __init__(self):
        super().__init__()
        self.collectors = []
        self.temporal = {}
        self.history = {}
        self.device_names = {}

    def on_mount(self) -> None:
        if not initialize():
            self.update("[red]Failed to initialize NVML[/red]")
            return

        device_count = get_device_count()
        if device_count == 0:
            self.update("[yellow]No GPUs found[/yellow]")
            return

        for i in range(device_count):
            try:
                collector = MetricsCollector(i)
                self.collectors.append((i, collector))
                self.temporal[i] = TemporalAnalyzer(window_size=60)
                self.history[i] = deque()

                info = get_gpu_info(i)
                self.device_names[i] = info.name if info else f"GPU {i}"
            except RuntimeError:
                pass

        if self.collectors:
            self.set_interval(UPDATE_INTERVAL, self.update_metrics)

    def update_metrics(self) -> None:
        """Update metrics for all GPUs"""
        if not self.collectors:
            return

        try:
            current_time = time()
            panels = []

            for device_index, collector in self.collectors:
                m = collector.collect()
                bottleneck = classify(m)

                # feed temporal analyzer
                self.temporal[device_index].update(m)
                patterns = self.temporal[device_index].detect()

                # get recommendations
                recs = recommend(bottleneck=bottleneck, patterns=patterns)

                # save to history
                self.history[device_index].append(MetricSnapshot(
                    timestamp=current_time,
                    temperature=m.temperature,
                    power_usage=m.power_usage or 0.0,
                    gpu_utilization=m.gpu_utilization,
                    memory_utilization=m.memory_utilization,
                    memory_used=m.memory_used,
                    gpu_clock=m.gpu_clock,
                    memory_clock=m.memory_clock,
                    bottleneck=bottleneck.bottleneck.value,
                ))
                self._clean_old_snapshots(device_index, current_time)

                # build display
                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="yellow")

                # utilization bars
                gpu_bar = _make_bar(m.gpu_utilization, 100)
                table.add_row("GPU Utilization", f"{gpu_bar} {m.gpu_utilization}%")

                mem_bar = _make_bar(m.memory_utilization, 100)
                table.add_row("Memory Controller", f"{mem_bar} {m.memory_utilization}%")

                # VRAM
                vram_bar = _make_bar(m.memory_used, m.memory_total)
                table.add_row(
                    "VRAM",
                    f"{vram_bar} {m.memory_used / (1024**3):.1f} / "
                    f"{m.memory_total / (1024**3):.1f} GB "
                    f"({m.memory_used_pct:.0f}%)",
                )

                # clocks
                clock_str = f"{m.gpu_clock} / {m.max_gpu_clock} MHz"
                if m.clock_reduction_pct > 1:
                    clock_str += f" ({m.clock_reduction_pct:.0f}% reduced)"
                table.add_row("Clocks", clock_str)

                # temperature
                table.add_row("Temperature", f"{m.temperature}C")

                # power
                if m.power_usage is not None:
                    if m.power_limit is not None:
                        power_bar = _make_bar(m.power_usage, m.power_limit)
                        power_str = f"{power_bar} {m.power_usage:.1f}W / {m.power_limit:.1f}W"
                    else:
                        power_str = f"{m.power_usage:.1f}W"
                    table.add_row("Power", power_str)

                # fan
                if m.fan_speed is not None:
                    fan_bar = _make_bar(m.fan_speed, 100)
                    table.add_row("Fan Speed", f"{fan_bar} {m.fan_speed}%")

                # throttle
                if m.throttle.is_throttled:
                    table.add_row("Throttle", f"[red]{m.throttle.summary}[/red]")
                else:
                    table.add_row("Throttle", f"[green]{m.throttle.summary}[/green]")

                # bottleneck status
                table.add_row("", "")
                color = _bottleneck_color(bottleneck.bottleneck.value)
                conf_pct = int(bottleneck.confidence * 100)
                table.add_row(
                    "Status",
                    f"[{color}]{bottleneck.bottleneck.value}[/{color}] "
                    f"[dim]({conf_pct}%)[/dim]",
                )
                table.add_row("", f"[dim]{bottleneck.detail}[/dim]")

                # warnings
                for w in bottleneck.warnings:
                    table.add_row("", f"[yellow]{w}[/yellow]")

                # temporal patterns
                for p in patterns:
                    pcolor = "red" if p.severity == "critical" else "yellow"
                    table.add_row("", f"[{pcolor}]{p.detail}[/{pcolor}]")

                # top recommendation
                if recs and recs[0].priority <= 2:
                    table.add_row("", "")
                    table.add_row(
                        "Action",
                        f"[bold]{recs[0].title}[/bold]",
                    )
                    for action in recs[0].actions[:2]:
                        table.add_row("", f"[dim]  - {action}[/dim]")

                device_name = self.device_names.get(device_index, f"GPU {device_index}")
                panel = Panel(table, title=f"{device_name}", border_style="green")
                panels.append(panel)

            group = Group(*panels)
            self.update(group)
        except Exception as e:
            self.update(f"[red]Error: {e}[/red]")

    def _clean_old_snapshots(self, device_index: int, current_time: float) -> None:
        """Remove snapshots older than PEAK_WINDOW seconds"""
        history = self.history.get(device_index)
        if not history:
            return
        while history and (current_time - history[0].timestamp) > PEAK_WINDOW:
            history.popleft()

    def _get_peaks(self, device_index: int, current_time: float) -> dict:
        """Get peak values from history"""
        self._clean_old_snapshots(device_index, current_time)
        history = self.history.get(device_index, [])

        if not history:
            return {}

        return {
            "temperature": max(s.temperature for s in history),
            "power_usage": max(s.power_usage for s in history),
            "gpu_utilization": max(s.gpu_utilization for s in history),
            "memory_utilization": max(s.memory_utilization for s in history),
            "memory_used": max(s.memory_used for s in history),
            "gpu_clock": max(s.gpu_clock for s in history),
            "memory_clock": max(s.memory_clock for s in history),
            "bottleneck": history[-1].bottleneck,
        }


class PeakMetrics(Static):
    """Display peak metrics from history"""

    def __init__(self, metrics_widget):
        super().__init__()
        self.metrics_widget = metrics_widget

    def on_mount(self) -> None:
        self.set_interval(UPDATE_INTERVAL, self.update_peaks)

    def update_peaks(self) -> None:
        """Update peak metrics display"""
        if not self.metrics_widget.collectors:
            self.update("[yellow]No peak data available[/yellow]")
            return

        try:
            current_time = time()
            panels = []

            for device_index, collector in self.metrics_widget.collectors:
                peaks = self.metrics_widget._get_peaks(device_index, current_time)

                if not peaks:
                    continue

                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("Metric", style="cyan")
                table.add_column("Peak Value", style="yellow")

                gpu_bar = _make_bar(peaks["gpu_utilization"], 100)
                table.add_row("GPU Utilization", f"{gpu_bar} {peaks['gpu_utilization']}%")

                mem_bar = _make_bar(peaks["memory_utilization"], 100)
                table.add_row("Memory Controller", f"{mem_bar} {peaks['memory_utilization']}%")

                table.add_row("Temperature", f"{peaks['temperature']:.0f}C")

                if peaks["power_usage"] > 0:
                    table.add_row("Power", f"{peaks['power_usage']:.1f}W")

                gpu_bar = _make_bar(peaks["gpu_clock"], 3000)
                table.add_row("GPU Clock", f"{peaks['gpu_clock']} MHz")
                table.add_row("Memory Clock", f"{peaks['memory_clock']} MHz")

                # current status
                if peaks.get("bottleneck"):
                    color = _bottleneck_color(peaks["bottleneck"])
                    table.add_row("", "")
                    table.add_row("Status", f"[{color}]{peaks['bottleneck']}[/{color}]")

                device_name = self.metrics_widget.device_names.get(
                    device_index, f"GPU {device_index}"
                )
                panel = Panel(
                    table,
                    title=f"{device_name} Peak Values (last 60s)",
                    border_style="yellow",
                )
                panels.append(panel)

            if panels:
                group = Group(*panels)
                self.update(group)
            else:
                self.update("[dim]No peak data yet[/dim]")
        except Exception as e:
            self.update(f"[red]Error: {e}[/red]")


def _bottleneck_color(bottleneck_type: str) -> str:
    """Get color for bottleneck type"""
    colors = {
        "compute_bound": "blue",
        "memory_bandwidth_bound": "cyan",
        "memory_capacity_bound": "bright_red",
        "thermal_throttled": "red",
        "power_limited": "yellow",
        "data_starved": "magenta",
        "balanced": "green",
        "idle": "dim",
        "unknown": "white",
    }
    return colors.get(bottleneck_type, "white")


class App(TextualApp):
    """NVSonar terminal interface"""

    TITLE = "NVSonar"
    SUB_TITLE = "GPU Diagnostic Tool"
    CSS = """
    DeviceList {
        height: auto;
        padding: 1;
        margin: 1;
    }

    LiveMetrics {
        height: auto;
        padding: 0;
        margin: 1;
    }

    TabbedContent {
        height: auto;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        metrics_widget = LiveMetrics()

        with TabbedContent():
            with TabPane("Overview", id="overview"):
                yield DeviceList()
                yield metrics_widget
            with TabPane("History", id="history"):
                yield PeakMetrics(metrics_widget)
        yield Footer()

    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()


if __name__ == "__main__":
    app = App()
    app.run()
