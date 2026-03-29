"""Command-line interface for NVSonar"""

import sys

from nvsonar import __version__
import typer

app = typer.Typer(add_completion=False, invoke_without_command=True)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print version and exit")
):
    """GPU diagnostic tool — run without arguments for TUI"""
    if version:
        typer.echo(f"nvsonar {__version__}")
        sys.exit(0)
    if ctx.invoked_subcommand is not None:
        return

    try:
        from nvsonar.tui.app import App

        tui_app = App()
        tui_app.run()
    except ImportError as e:
        typer.echo(f"Error: Failed to import TUI: {e}", err=True)
        typer.echo("Install dependencies: pip install nvsonar", err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


@app.command()
def report(
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
    csv: bool = typer.Option(False, "--csv", help="Output as CSV"),
    gpu: int = typer.Option(-1, "--gpu", help="GPU index, -1 for all"),
):
    """One-shot GPU diagnostic report"""
    from nvsonar.monitor import initialize, get_device_count, get_gpu_info, MetricsCollector
    from nvsonar.analysis import classify, detect_outliers, recommend
    from nvsonar.report import print_report, to_json, report_to_csv_row, to_csv

    if not initialize():
        typer.echo("Error: failed to initialize NVML, no NVIDIA GPU found", err=True)
        sys.exit(1)

    device_count = get_device_count()
    if device_count == 0:
        typer.echo("Error: no GPUs detected", err=True)
        sys.exit(1)

    # which GPUs to report on
    if gpu >= 0:
        if gpu >= device_count:
            typer.echo(f"Error: GPU {gpu} not found, {device_count} available", err=True)
            sys.exit(1)
        indices = [gpu]
    else:
        indices = list(range(device_count))

    # collect metrics for all requested GPUs
    all_metrics = {}
    for i in indices:
        collector = MetricsCollector(i)
        all_metrics[i] = collector.collect()

    # run outlier detection if multiple GPUs
    outliers = []
    if len(all_metrics) > 1:
        outliers = detect_outliers(all_metrics)

    # report per GPU
    json_reports = []
    csv_rows = []
    for i in indices:
        info = get_gpu_info(i)
        if not info:
            continue

        metrics = all_metrics[i]
        bottleneck = classify(metrics)

        # outliers for this GPU
        gpu_outliers = [o for o in outliers if o.gpu_index == i]
        recs = recommend(bottleneck=bottleneck, outliers=gpu_outliers)

        # save to history
        from nvsonar.history import save_from_metrics
        save_from_metrics(i, info.name, metrics, bottleneck)

        if json:
            json_reports.append(to_json(info, metrics, bottleneck, recommendations=recs))
        elif csv:
            csv_rows.append(report_to_csv_row(info, metrics, bottleneck))
        else:
            print_report(info, metrics, bottleneck, recommendations=recs)

    if json:
        if len(json_reports) == 1:
            typer.echo(json_reports[0])
        else:
            typer.echo("[" + ",\n".join(json_reports) + "]")
    elif csv:
        typer.echo(to_csv(csv_rows))


@app.command()
def history(
    gpu: int = typer.Option(-1, "--gpu", help="GPU index, -1 for all"),
    days: int = typer.Option(7, "--days", help="Number of days to show"),
):
    """Show GPU health trends over time"""
    from nvsonar.history import print_history

    gpu_index = gpu if gpu >= 0 else None
    print_history(gpu_index=gpu_index, days=days)


@app.command()
def benchmark(
    memory: bool = typer.Option(False, "--memory", help="Run memory bandwidth only"),
    compute: bool = typer.Option(False, "--compute", help="Run compute throughput only"),
    pcie: bool = typer.Option(False, "--pcie", help="Run PCIe bandwidth only"),
):
    """Run GPU performance benchmarks"""
    from nvsonar.monitor import initialize, get_gpu_info
    from nvsonar.baselines.specs import find_specs

    if not initialize():
        typer.echo("Error: failed to initialize NVML, no NVIDIA GPU found", err=True)
        sys.exit(1)

    info = get_gpu_info(0)
    if not info:
        typer.echo("Error: could not read GPU info", err=True)
        sys.exit(1)

    specs = find_specs(info.name)

    typer.echo(f"GPU: {info.name}")
    typer.echo(f"Compiling benchmarks (first run may take a few seconds)...")
    typer.echo()

    run_all = not memory and not compute and not pcie

    if memory or run_all:
        try:
            from nvsonar.benchmark import run_memory
            result = run_memory()
            typer.echo(f"Memory bandwidth:")
            typer.echo(f"  Read:  {result.read_gbps:.1f} GB/s")
            typer.echo(f"  Write: {result.write_gbps:.1f} GB/s")
            typer.echo(f"  Copy:  {result.copy_gbps:.1f} GB/s")
            if specs:
                pct = (result.copy_gbps / specs.memory_bandwidth_gbps) * 100
                typer.echo(f"  Spec:  {specs.memory_bandwidth_gbps:.0f} GB/s ({pct:.0f}% of theoretical)")
            typer.echo()
        except RuntimeError as e:
            typer.echo(f"Memory benchmark failed: {e}", err=True)

    if compute or run_all:
        try:
            from nvsonar.benchmark import run_compute
            result = run_compute()
            typer.echo(f"Compute throughput:")
            typer.echo(f"  FP32:  {result.tflops:.2f} TFLOPS")
            if specs:
                pct = (result.tflops / specs.fp32_tflops) * 100
                typer.echo(f"  Spec:  {specs.fp32_tflops:.1f} TFLOPS ({pct:.0f}% of theoretical)")
            typer.echo()
        except RuntimeError as e:
            typer.echo(f"Compute benchmark failed: {e}", err=True)

    if pcie or run_all:
        try:
            from nvsonar.benchmark import run_pcie
            result = run_pcie()
            typer.echo(f"PCIe bandwidth:")
            typer.echo(f"  Host->GPU: {result.h2d_gbps:.1f} GB/s")
            typer.echo(f"  GPU->Host: {result.d2h_gbps:.1f} GB/s")
            if specs:
                pct = (result.h2d_gbps / specs.pcie_bandwidth_gbps) * 100
                typer.echo(f"  Spec:      {specs.pcie_bandwidth_gbps:.1f} GB/s Gen{specs.pcie_gen} ({pct:.0f}% of theoretical)")
            typer.echo()
        except RuntimeError as e:
            typer.echo(f"PCIe benchmark failed: {e}", err=True)


if __name__ == "__main__":
    app()
