FROM python:3.11-slim-bookworm

# Open3d requires special packages
RUN apt-get update && apt-get install libgomp1 ffmpeg libsm6 libxext6 libx11-6 -y

# Install poetry
RUN pip install "poetry==1.8.3"

# Move the files into the container
WORKDIR /project
COPY . /project

# Install dependencies
RUN poetry env use python3.11
RUN poetry install --no-interaction --no-cache

# Run the script
CMD poetry run python data/download_dataset.py && poetry run python examples/0_downsample_stl.py && poetry run python examples/1_run_pointcloud_registration.py && poetry run python examples/2_cluster_data.py