FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only necessary files first for better caching
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir .

# Create non-root user for security
RUN useradd -m -s /bin/bash simpulse && \
    chown -R simpulse:simpulse /app

# Switch to non-root user
USER simpulse

# Set the entrypoint
ENTRYPOINT ["simpulse"]

# Default command shows help
CMD ["--help"]