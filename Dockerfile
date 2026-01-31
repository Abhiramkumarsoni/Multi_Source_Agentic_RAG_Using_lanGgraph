# Use Python 3.11 slim image (smaller and faster)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory for user
WORKDIR $HOME/app

# Copy requirements first (for Docker caching - faster builds)
COPY --chown=user requirements_hf.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Expose the port Streamlit runs on
EXPOSE 7860

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check to verify app is running
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_ui/app_final.py", "--server.port=7860", "--server.address=0.0.0.0"]
