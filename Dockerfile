FROM python:3.10

WORKDIR /app

# copy only needed files (cleaner build)
COPY . /app

# install dependencies
RUN pip install --no-cache-dir fastapi uvicorn openenv

# Hugging Face port
EXPOSE 7860

# run app
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]