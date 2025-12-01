FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 1) Basics
RUN apt-get update && apt-get install -y \
    git python3 python3-pip ffmpeg libgl1 curl && \
    rm -rf /var/lib/apt/lists/*

# 2) Create app folder & copy files
WORKDIR /workspace
COPY . .

# 3) Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 4) Expose FastAPI server port
EXPOSE 8000

# 5) Start command
CMD ["python3", "server.py"]
