# Filename: Dockerfile

# 1. Select a multi-platform base image (supports arm64 and amd64)
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Set default environment variables
# (Your run.sh script will use these)
ENV N_SAMPLES="500"
ENV PROVIDER="deepseek"

# 4. Declare a build-time argument for the API key
# This avoids hardcoding your secret key into the image
ARG DEEPSEEK_API_KEY
ENV DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}

# 5. Copy the dependencies file
COPY requirements.txt .

# 6. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy all your project files into the image
# (This will include your iems490/ folder, run.sh, and python scripts)
COPY . .

# 8. Give the run.sh script execution permissions
RUN chmod +x run.sh

# 9. Set the default command to execute when the container starts
CMD ["./run.sh"]