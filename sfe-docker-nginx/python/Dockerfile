FROM python:3.10-slim

WORKDIR /app

# Installer graphviz pour les arbres XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir pydot

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 --reload --timeout-keep-alive 300"]
