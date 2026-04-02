FROM python:3.10

WORKDIR /app

# copy only needed files (cleaner build)
COPY . /app

# install runtime dependencies required by the environment server
RUN pip install --no-cache-dir "openenv-core>=0.2.2" fastapi uvicorn openai

# Hugging Face port
EXPOSE 7860

# run app
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
