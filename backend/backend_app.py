from ec2_functions import create_ec2_instance, send_cmd_pem
from fastapi import FastAPI
import requests
import random
import time
import json

app = FastAPI()

@app.post("/deploy_rag/")
def deploy_rag(data: dict):
  create_ec2 = data.get('ec2', True)
  if create_ec2:
    region_name = data.get('region', 'ap-south-1')
    access_key = data['access_key']
    secret_access_key = data['secret_access_key']
    ebs_volume = data.get('ebs_volume', 28)
    instance_type = data.get('instance_type', 't2.large')
    instance_category = data.get('instance_category', 'spot')
    instance_ami_id = data.get('instance_ami_id', 'ami-007020fd9c84e18c7')

    instance_id, public_ip, public_dns, pkey_path = create_ec2_instance({
      "region_name":region_name, 
      "access_key":access_key, 
      "secret_access_key":secret_access_key,
      "ebs_volume":ebs_volume, 
      "instance_type":instance_type, 
      "instance_category":instance_category,
      "instance_name": "deployed_llm_"+str(random.randint(1, 10000)),
      "instance_ami_id": instance_ami_id
      })
  else:
    public_ip = data.get("public_ip")
    pkey_path = data.get("pkey_path")
  
  print('ec2 instance created successfully')
  ## Send commands to start the rag system with necessary settings
  username = 'ubuntu' # By default for ec2
  command_list = [
    "sudo apt-get -y update && sudo apt-get -y upgrade",  
    "sudo apt-get install -y python3-pip && sudo apt-get install -y tmux",
    "sudo apt install -y docker.io",
    "sudo systemctl start docker && sudo systemctl enable docker",
    "sudo tmux new -d -s rag_session",
    "sudo tmux send-keys -t rag_session 'sudo docker run -p 80:8080 ratna1sambhav/ai_tuners_rag_system:0.0' Enter",
    # sudo docker run -d -p 80:8080 ratna1sambhav/ai_tuners_rag_system:0.0
    ]
  stdout = send_cmd_pem(public_ip, username, pkey_path, command_list)
  # time.sleep()  
  # requests.post(f'http://{public_dns}:80/settings/', json={"palm_api_key": "AIzaSyD-gUGR1747OmPBrTEBk2dJBo2yBLzlBQ8"})
  return {"Logs": stdout,
  "Instructions": f"Your API has been deployed on the ec2 instance. Use {public_dns}:80 to send your requests.",
  "More instructions": "Use /connects3/ endpoint to send your s3 bucket.",
  "Parameters for /connects3/ endpoint": {
    "bucket_name": "s3 bucket name",
    "access_key": "access_key_id and secret access key of the IAM Role that has full access to the s3 bucket",
    "folder_name": "name of the folder inside the s3 bucket."
  },
  "Even more instructions": "Ask your question on /makequery/ endpoint with {'question': 'your question'} format"
}


@app.get("/fine_tune_model/")
def fine_tune(data: dict):
  create_ec2 = data.get('ec2', True)
  if create_ec2:
    region_name = data.get('region', 'ap-south-1')
    access_key = data['access_key']
    secret_access_key = data['secret_access_key']
    ebs_volume = data.get('ec2_volume', 28)
    instance_type = data.get('instance_type', 't2.micro')
    instance_category = data.get('instance_category', 'spot')
    instance_ami_id = data.get('instance_ami_id', 'ami-007020fd9c84e18c7')

    instance_id, public_ip, public_dns, pkey_path = create_ec2_instance({
      "region_name":region_name, 
      "access_key":access_key, 
      "secret_access_key":secret_access_key,
      "ebs_volume":ebs_volume, 
      "instance_type":instance_type, 
      "instance_category":instance_category,
      "instance_name": "fine_tune_"+str(random.randint(1, 10000)),
      "instance_ami_id": instance_ami_id
      })
  else:
    public_ip = data.get("public_ip")
    pkey_path = data.get("pkey_path")
  
  ## Send commands to setup the instance and start fine-tuning
  with open('./install_cuda.sh', 'r') as f:
    cuda_driver_commands_txt = f.read()
    f.close()
    
  json_train_data = json.dumps(data.get('training_info'))
  wandb_api_key = data.get('WANDB_API_KEY', '')
  username = 'ubuntu' # By default for ec2
  command_list = [
    "sudo apt-get -y update && sudo apt-get -y upgrade",  
    "sudo apt-get install -y python3-pip",
    "sudo apt install -y docker.io",
    "sudo systemctl start docker && sudo systemctl enable docker",
    f"echo '{json_train_data}' > ./prompt_dir/prompt.json",
    "tmux new -d -s fine_tune_session",
    # Install nvidia drivers
    f"tmux send-keys -t fine_tune_session  'echo {cuda_driver_commands_txt} > ./cuda_driver_install.sh' Enter",
    # "tmux send-keys -t fine_tune_session 'chmod +x ./cuda_driver_install.sh' Enter",
    # "tmux send-keys -t fine_tune_session './cuda_driver_install.sh' Enter",
    f"tmux send-keys -t fine_tune_session 'sudo docker run -e WANDB_API_KEY={wandb_api_key} -v $(pwd)/model_params/:/tuning_app/ -v $(pwd)/prompt_dir/:/prompt_dir/ ratna1sambhav/ai_tuners_axolotl_ft:0.1' Enter",
    ]
  #'sudo docker run -v $(pwd):/tuning_app/ ai_tuners_fine_tune_axolotl'
  stdout = send_cmd_pem(public_ip, username, pkey_path, command_list)  
  
  return {"Logs": stdout, "Instructions": f"Model Training has been initiated. Please login to your instance and run 'tmux ls' and then attach to the tmux session"}
