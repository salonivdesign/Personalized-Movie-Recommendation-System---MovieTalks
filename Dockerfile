# UNTESTED template (Docker is not available in the environment this project was built in).
# Training happens at image BUILD time, never at app start.
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python scripts/download_data.py && python scripts/train.py --no-tune

# Hugging Face Docker Spaces expect 7860; most other hosts inject $PORT.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
