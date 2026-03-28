# Changelog

## [2.0.0] - 2026-03-27

### Added
- Analysis layer: bottleneck classification (compute, memory bandwidth, memory capacity, power, thermal, data-starved)
- Temporal pattern detection (clock oscillation, temperature trends, utilization dips, memory creep)
- Multi-GPU outlier detection via Z-scores
- Actionable recommendations engine with priorities
- Report command: `nvsonar report` with Rich terminal output
- JSON report output: `nvsonar report --json`
- GPU selection: `nvsonar report --gpu N`
- Health scoring (0-100) with letter grades (A-F)
- Throttle bitmask decoder with severity levels
- PCIe link degradation detection
- ECC error monitoring

### Changed
- Complete architecture rewrite: monitor/ -> analysis/ -> report/ layers
- TUI rewritten to use new analysis layer
- CLI uses typer subcommands (nvsonar for TUI, nvsonar report for diagnostics)

### Removed
- Old core/ module (replaced by analysis/)
- Old utils/ module (replaced by monitor/hardware)
- Old single-threshold bottleneck detection

## [1.1.0] - 2026-02-11

### Added
- Bottleneck types
- Utilization tracking
- Peak value history tab (60 sec window)
- Visuals
- Tabbed interface (Overview/History/Settings)

## [1.0.2] - 2026-01-30

### Fixed
- Personal link

## [1.0.1] - 2026-01-30

### Added
- Multi-GPU support

### Changed
- Simplified to use pynvml directly
- Removed unimplemented feature claims from documentation

## [1.0.0] - 2026-01-28

### Added
- GPU detection and information gathering
- Real-time metrics monitoring (temperature, power, utilization, clocks)
- CLI interface with `list`, `monitor`, and `tui` commands
- Interactive terminal UI with live metrics display
- Graceful handling of unsupported GPU features
- Test suite with GPU-specific markers

### Fixed
- CI workflow configuration for PyPI publishing

## [0.0.1] - 2026-01-27

### Added
- Initial project structure
- PyPI publishing setup with GitHub Actions
- Apache 2.0 license
- Code of conduct and security policy
