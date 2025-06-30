# Simpulse Production Container
FROM ubuntu:22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Lean 4 and Lake via elan
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:stable
ENV PATH="/root/.elan/bin:$PATH"

# Verify Lean installation
RUN lean --version && lake --version

# Install Claude Code CLI (if available)
# Note: This would require Claude Code to be publicly available
# For now, we'll install dependencies and handle the CLI check at runtime
RUN pip3 install --no-cache-dir PyGithub plotly pandas psutil

# Set up Python environment
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip3 install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY .github/ ./.github/

# Install Simpulse in development mode
RUN pip3 install --no-cache-dir -e .

# Create directories for runtime data
RUN mkdir -p /app/workspace /app/reports /app/cache /app/logs

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables
ENV SIMPULSE_CACHE_DIR=/app/cache
ENV SIMPULSE_LOG_DIR=/app/logs
ENV SIMPULSE_WORKSPACE_DIR=/app/workspace
ENV SIMPULSE_REPORTS_DIR=/app/reports

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import simpulse; print('OK')" || exit 1

# Default entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["--help"]