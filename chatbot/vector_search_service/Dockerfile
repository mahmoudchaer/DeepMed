FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (app.py, init_db.py, docs/, etc.)
COPY . .

# Initialize the DB on container startup
ENV CHROMAPERSISTDIR=./chroma_data

EXPOSE 5202

CMD python3 init_db.py && uvicorn app:app --host 0.0.0.0 --port 5202
