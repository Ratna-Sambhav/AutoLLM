FROM ubuntu

# Set the working directory in the container
WORKDIR /tuning_app

# Copy the current directory contents into the container at /tuning_app
COPY . /tuning_app/

# Volumes for training logs and final model weights
VOLUME /tuning_app/results/logs 
VOLUME /tuning_app/model_weights

# Install any needed packages specified in requirements.txt
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y python3-pip 
RUN pip3 install -q git+https://github.com/huggingface/transformers && pip3 install torch accelerate peft bitsandbytes trl

# Run fine_tuning_pipeline.py when the container launches
CMD ["python3", "fine_tuning_pipeline.py", "--config_path", "/tuning_app/config.json"]
