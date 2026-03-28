"""Command-line interface for NVSonar"""

import sys

import typer

app = typer.Typer(add_completion=False, invoke_without_command=True)


@app.callback()
def main(ctx: typer.Context):
    """GPU diagnostic tool — run without arguments for TUI"""
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


if __name__ == "__main__":
    app()
