# Use a lightweight base image
FROM python:3.11-slim

# set workdir
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# copy application code
COPY src .
COPY eval_data data

# expose port
EXPOSE 8000

# launch with multiple workers for high concurrency
# using uvicorn's built-in worker support
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4"]