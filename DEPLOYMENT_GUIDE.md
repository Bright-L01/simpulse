# Simpulse Deployment Guide 🚀

## Overview
Simpulse is a lightning-fast optimizer for Lean 4 simp rules that reduces proof search time by 2.83x. It's designed to be simple to deploy and use.

## Installation Methods

### 1. Docker (Recommended for CI/CD)

Pull and run the pre-built image:
```bash
docker pull ghcr.io/bright-l01/simpulse:latest
docker run --rm -v $(pwd):/work ghcr.io/bright-l01/simpulse optimize /work
```

Build from source:
```bash
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse
docker build -t simpulse .

# Run optimization
docker run --rm -v $(pwd):/work simpulse optimize /work

# Run with apply
docker run --rm -v $(pwd):/work simpulse optimize --apply /work

# Health check
docker run --rm simpulse --health
```

### 2. Pip Installation

From PyPI (when published):
```bash
pip install simpulse
simpulse --health
simpulse optimize .
```

From GitHub:
```bash
pip install git+https://github.com/Bright-L01/simpulse.git
simpulse --health
```

### 3. Development Installation

Clone and install in editable mode:
```bash
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse
pip install -e .
simpulse --health
```

Using Poetry:
```bash
git clone https://github.com/Bright-L01/simpulse.git
cd simpulse
poetry install
poetry run simpulse --health
```

## Usage Examples

### Basic Usage

Check if optimization would help:
```bash
simpulse check path/to/lean/project
```

Preview optimizations:
```bash
simpulse optimize path/to/lean/project
```

Apply optimizations:
```bash
simpulse optimize --apply path/to/lean/project
```

### Advanced Usage

Save optimization plan:
```bash
simpulse optimize --output optimizations.json path/to/project
```

Debug mode for troubleshooting:
```bash
simpulse --debug optimize path/to/project
```

JSON output for CI integration:
```bash
simpulse optimize --json path/to/project
```

### Docker Usage

Basic optimization:
```bash
docker run --rm -v $(pwd):/work simpulse optimize /work
```

With environment variables:
```bash
docker run --rm \
  -e SIMPULSE_MAX_FILE_SIZE=2000000 \
  -e SIMPULSE_TIMEOUT=60 \
  -v $(pwd):/work \
  simpulse optimize /work
```

Interactive shell:
```bash
docker run --rm -it -v $(pwd):/work simpulse /bin/bash
```

## Configuration

### Environment Variables

```bash
# Lean executable path (default: "lean")
export LEAN_PATH=/path/to/lean

# Lake build tool path (default: "lake") 
export LAKE_PATH=/path/to/lake

# Safety limits
export SIMPULSE_MAX_FILE_SIZE=1000000    # 1MB limit
export SIMPULSE_TIMEOUT=30               # 30 second timeout
export SIMPULSE_MAX_MEMORY=1000000000    # 1GB memory limit
```

### .env File

Create a `.env` file in your project:
```env
LEAN_PATH=lean
LAKE_PATH=lake
SIMPULSE_MAX_FILE_SIZE=2000000
SIMPULSE_TIMEOUT=60
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Optimize Lean Code
on: [push, pull_request]

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Simpulse
        run: |
          docker run --rm -v ${{ github.workspace }}:/work \
            ghcr.io/bright-l01/simpulse:latest \
            optimize --json /work > optimization-report.json
      
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: optimization-report
          path: optimization-report.json
```

### GitLab CI

```yaml
optimize:
  image: ghcr.io/bright-l01/simpulse:latest
  script:
    - simpulse --health
    - simpulse optimize --apply .
  artifacts:
    paths:
      - "**/*.lean"
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: simpulse-check
        name: Check simp optimizations
        entry: simpulse check
        language: system
        files: '\.lean$'
        pass_filenames: false
```

## Health Checks

Verify installation:
```bash
simpulse --health
```

Expected output:
```
✅ Health check passed
  - Optimizer: OK
  - File processing: OK
  - Lean path: lean
```

For Docker:
```bash
docker run --rm simpulse --health
```

## Troubleshooting

### Common Issues

1. **"File too large" errors**
   - Increase limit: `export SIMPULSE_MAX_FILE_SIZE=5000000`
   - Or split large files

2. **"Timeout" errors**
   - Increase timeout: `export SIMPULSE_TIMEOUT=120`
   - Or optimize smaller directories

3. **"Command not found"**
   - Ensure simpulse is in PATH
   - Or use `python -m simpulse`

4. **Permission errors in Docker**
   - Use `--user $(id -u):$(id -g)` flag
   - Or ensure files have correct permissions

### Debug Mode

Enable detailed logging:
```bash
simpulse --debug optimize .
```

## Performance Tips

1. **Optimize incrementally**: Run on specific directories rather than entire projects
2. **Use --json output**: Parse results programmatically for large projects
3. **Set appropriate limits**: Adjust timeouts and file size limits for your project
4. **Regular optimization**: Run after major refactoring or before releases

## Requirements

- Python 3.10 or higher
- Lean 4 project with simp rules
- No external dependencies beyond Python stdlib + click + rich

## Support

- Issues: https://github.com/Bright-L01/simpulse/issues
- Documentation: https://github.com/Bright-L01/simpulse
- Email: brightliu@college.harvard.edu

## License

MIT License - see LICENSE file for details.