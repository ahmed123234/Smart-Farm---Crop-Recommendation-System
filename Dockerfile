FROM agrigorev/zoomcamp-model:2025


# 1. Install uv (already in cache, but keeping it)
RUN pip install uv

# 2. CREATE the virtual environment inside the container
RUN uv venv

# 3. Copy pyproject.toml
COPY pyproject.toml /code

# 4. Install dependencies into the VENV
# We use a shell command to activate the venv's context (similar to 'source .venv/bin/activate')
# and then run the install command.
RUN . .venv/bin/activate && uv pip install .[dev]

# 5. Copy your FastAPI script
COPY app.py /code

EXPOSE 8000

# 6. Run the application from the VENV's python executable
# We must use the Python interpreter from the virtual environment:
CMD ["/code/.venv/bin/python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
