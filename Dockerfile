FROM vllm/vllm-openai:v0.9.1

# Clear the default entrypoint
ENTRYPOINT []

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8

# Upgrade system packages
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Fully replace blinker, ignoring the broken old installation
RUN pip install --upgrade pip \
    && pip install --ignore-installed blinker

# Copy files
COPY requirements.txt .
COPY ./setup.py .
COPY ./dots_ocr ./dots_ocr
RUN pip install -e . \
    && rm -rf /root/.cache/pip
COPY ./start.sh .
RUN chmod +x start.sh
COPY ./test_images_dir ./test_images_dir

# Expose port
EXPOSE 8501
#EXPOSE 7860 # gradio app

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll
ENV STREAMLIT_SERVER_RUNON_SAVE=true

# Healthcheck (for streamlit app, ensure curl is available)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run app
CMD ["./start.sh"]
