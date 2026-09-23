# Production image for the Wine Quality Classifier API.
# Built only after tests + data validation + model validation
# pass in CI (see params.yaml validation gate).

FROM python:3.11-slim

WORKDIR /app

# Install production dependencies only (not requirements-dev.txt -
# no pytest/ruff/dvc needed inside the running container)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY src/ ./src/
COPY params.yaml .

# Copy the trained model artifact (produced by `python -m src.train`
# before this image is built - see CI Stage 10 in project spec)
COPY models/model.pkl ./models/model.pkl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]