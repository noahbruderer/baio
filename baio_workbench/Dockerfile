# We will be working in ubuntu 22.04
FROM ubuntu:22.04

# Install system packages
RUN apt-get update && apt-get install -y \
    nano \
    python3 \
    python3-pip \
    git \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /usr/src/app for the baio app
WORKDIR /usr/src/app

# Copy the requirements.txt file and install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the 'baio' directory into the container
# COPY baio baio/

# Copy the shell script into the container and make sure it is executable
# COPY startup.sh .
# RUN chmod +x ./startup.sh

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run the shell script when the container launches
# CMD ["./startup.sh"]

# Run an infinite loop
CMD tail -f /dev/null