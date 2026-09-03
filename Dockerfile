FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY . .
RUN mkdir -p /app/data/models /app/data/reports
EXPOSE 8000
CMD ["uvicorn", "aegisml.main:app", "--host", "0.0.0.0", "--port", "8000"]

