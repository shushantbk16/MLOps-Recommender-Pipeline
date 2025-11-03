# --- Use the highly stable Python 3.9 base image ---
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies (required for all C extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install our Python requirements
COPY requirements.txt .
# This single command will now work reliably on Python 3.9
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files and models
COPY demo.py .
COPY data/models/ ./data/models/

# Expose the port Streamlit runs on
EXPOSE 8501

# The command to run when the container starts
CMD ["streamlit", "run", "demo.py", "--server.port=8501", "--server.address=0.0.0.0"]