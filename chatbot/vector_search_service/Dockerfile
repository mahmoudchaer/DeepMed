FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .  # includes app.py, init_db.py, docs/, etc.

# Initialize the DB on container startup
ENV CHROMA_PERSIST_DIR=./chroma_data
ENV OPENAI_API_KEY=

EXPOSE 5202

CMD python3 init_db.py && uvicorn app:app --host 0.0.0.0 --port 5202
