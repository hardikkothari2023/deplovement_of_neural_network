# Use the official Python base image
FROM python:3.10.14-bookworm

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the entire project directory into the container
COPY . .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the entrypoint to run the training script and then start an interactive shell
ENTRYPOINT ["sh", "-c", "python train_pipeline.py && while true; do sleep 1000; done"]


#First Run this command "sudo docker build -t xor-model:v1 . 

#then use this command "docker run -it xor-model:v1" this will run the train_pipeline.py file which is return in the docker file .

#Then Run this command "sudo docker run -dit <container-id>this will help in running the predict.py file by using the below command .

#Then if you want to run the other command the use "docker exec -it <container-id> /bin/sh" then write python predict.py to run the predict file.

