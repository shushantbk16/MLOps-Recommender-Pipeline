# Stage 1: Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies
# We need build-essential for LightFM
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a requirements.txt file for our Python libraries
# We're doing this *before* copying our project code for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy all our project files into the container
# .dockerignore will handle what to skip
COPY . .

# --- DVC STEP ---
# This is where the magic happens. We pull the *real* models.
# This command pulls all files tracked by DVC (like our .pkl models)
# from your remote storage (e.g., Google Drive)
RUN dvc pull

# Expose the port Streamlit runs on
EXPOSE 8501

# The command to run when the container starts
CMD ["streamlit", "run", "demo.py", "--server.port=8501", "--server.address=0.0.0.0"]