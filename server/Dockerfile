FROM python:3.14-slim-bookworm

# Set working directory to the root where Hugging Face drops our files
WORKDIR /app

# System dependencies for UV
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy everything into the working directory
COPY . /app

# Completely wipe any copied Windows dependencies to prevent Path mismatch
RUN rm -rf /app/.venv

# Generate the virtual environment directly inside /app/.venv
RUN if [ -f uv.lock ]; then \
        uv sync --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

# Force inject uvicorn and security-pinned dependencies explicitly
RUN . /app/.venv/bin/activate && \
    pip install --no-cache-dir --upgrade \
        "uvicorn[standard]" \
        "fastmcp>=3.2.0" \
        "gradio>=6.11.0" \
        "setuptools>=82.0.1" && \
    rm -rf /root/.cache

# Least privilege security
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Lock the PATH specifically to our new Linux-native venv
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Exactly target the Hugging Face desired Entrypoint Array
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
