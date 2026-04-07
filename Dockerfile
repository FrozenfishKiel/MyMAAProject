FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["python", "src/main.py", "--config", "src/task-config/sample.yaml", "--dry-run"]

