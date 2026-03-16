FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by ChromaDB and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port Railway will route to
EXPOSE 8000

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

**New file: `.dockerignore`** (project root)
```
venv/
__pycache__/
*.pyc
*.pyo
chroma_db/
.env
.git/
*.zip