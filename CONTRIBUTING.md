# Contributing to NVSonar

## 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/<YOUR-USERNAME>/nvsonar.git
cd nvsonar
```

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## 3. Run

```bash
nvsonar              # TUI
nvsonar report       # one-shot report
nvsonar report --json
nvsonar report --csv
```

## 4. Code Style

We use black and isort for formatting:

```bash
make format   # auto-format
make lint     # check formatting
```

## 5. Tests

```bash
make test
```

## 6. Submit

1. Create a branch from main
2. Make your changes
3. Run `make format` and `make lint`
4. Push and open a pull request
