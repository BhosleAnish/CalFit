# ============================================================
# Dockerfile for NutritionApp (Flask + Tesseract + pyzbar)
# ============================================================

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────
# tesseract-ocr  : required by pytesseract (OCR fallback)
# libzbar0       : required by pyzbar (barcode scanning)
# libgl1 / libglib2.0-0 : required by Pillow / OpenCV
# ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────
# Copy requirements first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App source code ──────────────────────────────────────────
COPY . .

# ── Runtime directories ──────────────────────────────────────
RUN mkdir -p static/uploads .flask_session

# ── Tesseract path fix ───────────────────────────────────────
# process_label.py hard-codes the Windows path; override it
# via environment variable so pytesseract picks up the Linux binary.
ENV TESSERACT_CMD=/usr/bin/tesseract

# ── Environment defaults (override these at runtime) ─────────
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# ── Port ─────────────────────────────────────────────────────
EXPOSE 5000

# ── Entrypoint ───────────────────────────────────────────────
# Use gunicorn for production; falls back gracefully on Render.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]