# NVSonar

[![PyPI version](https://img.shields.io/pypi/v/nvsonar.svg)](https://pypi.org/project/nvsonar/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/nvsonar)](https://pepy.tech/project/nvsonar)

Active GPU diagnostic tool that identifies performance bottlenecks, detects anomalous patterns over time, and gives you actionable recommendations.

GPU monitoring tools show utilization percentages, but this can be misleading. A GPU reporting 100% utilization may actually be computing useful work, or wastefully stalled waiting on memory transfers, thermal throttling, or power limits. NVSonar analyzes real-time patterns from NVML metrics to identify what's actually limiting your GPU performance.

## Features

- Bottleneck classification (compute-bound, memory-bandwidth-bound, memory-capacity-bound, power-limited, thermal-throttled, data-starved)
- Temporal pattern detection (clock oscillation, temperature trends, utilization dips, memory leaks)
- Multi-GPU outlier detection via Z-scores
- Health scoring with A-F grades (0-100 per GPU)
- Throttle bitmask decoder with severity levels
- PCIe link degradation and ECC error monitoring
- Actionable recommendations with specific commands
- JSON output for automation and scripting

## Example

```
╭──── GPU 0: NVIDIA GeForce RTX 4090    Health: B (82/100) ─────╮
│  GPU utilization       95%                                    │
│  Memory controller     45%                                    │
│  VRAM                  18432MB / 24576MB (75%)                │
│  Clocks                2520 / 2520 MHz                        │
│  Temperature           78C                                    │
│  Power                 380W / 450W (84%)                      │
│  PCIe                  Gen4 x16                               │
│  Throttle              Clock state: Software Power Cap        │
│                                                               │
│  Bottleneck: power_limited (90% confidence)                   │
│  Power draw at 84% of limit, clocks reduced 0%                │
│                                                               │
│  Recommendations:                                             │
│    [P1] GPU is power limited                                  │
│      - Raise power limit: nvidia-smi -pl <watts>              │
│      - Check that all GPU power cables are connected          │
╰───────────────────────────────────────────────────────────────╯
```

## Requirements

- Python 3.10+
- NVIDIA GPU with driver installed
- Linux

## Installation and Usage

```bash
pip install nvsonar
```

```bash
nvsonar                  # interactive TUI
nvsonar report           # one-shot diagnostic
nvsonar report --json    # structured output for scripts/LLMs
nvsonar report --csv     # CSV output for spreadsheets
nvsonar report --gpu 0   # specific GPU
```

## License

Apache License 2.0

## Author

[Bekmukhamed Tursunbayev](https://btursunbayev.com)
