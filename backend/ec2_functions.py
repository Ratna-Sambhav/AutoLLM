import random
import boto3
import paramiko
import time

def create_ec2_instance(config_json):
  region_name = config_json.get('region', 'ap-south-1')
  access_key = config_json['access_key']
  secret_access_key = config_json['secret_access_key']
  ebs_volume = config_json.get('ebs_volume', 8)
  instance_type = config_json.get('instance_type', 't2.micro')
  instance_category = config_json.get('instance_category', 'spot')
  instance_name = config_json.get('instance_name', 'my_instance'+ str(random.randint(1, 10000)))

  # Define ec2 client and resource
  ec2_client = boto3.client('ec2', region_name=region_name, aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)
  ec2_resource = boto3.resource('ec2', region_name=region_name, aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)

  ## SECURITY GROUP
  # Create security group
  sg_name = 'sec_gr_at_' + str(random.randint(1, 10000))
  response = ec2_client.create_security_group(
      Description='My security group',
      GroupName=sg_name,
  )
  GroupId = response['GroupId']
  # Add inbound rules to security group made (outbound rules will be default)
  ip_permissions = [{
      # For Fastapi
      'IpProtocol': 'tcp',
      'FromPort': 3000,
      'ToPort': 3000,
      'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
      },
      {
      # For fastapi rag
      'IpProtocol': 'tcp',
      'FromPort': 80,
      'ToPort': 80,
      'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
      },
      {
      # For ssh
      'IpProtocol': 'tcp',
      'FromPort': 22,
      'ToPort': 22,
      'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
      },
  ]
  sc_group = ec2_client.authorize_security_group_ingress(
      GroupId=GroupId,
      IpPermissions=ip_permissions
  )

  ## EC2 KEY-PAIR
  # Creating key-pair and download .pem for ssh connection
  keypair_name = 'at_programmatic_access_' + str(random.randint(1, 100000))
  new_keypair = ec2_client.create_key_pair(KeyName=keypair_name)
  
  pkey_path = './my_key.pem'
  with open(pkey_path, 'w') as file:
      file.write(new_keypair['KeyMaterial'])


  ## EC2 Creation
  # instance_name = 'rag_application' + str(random.randint(1, 10000))
  # Create an ec2 instance of necessary configuration
  if instance_category == 'spot':
    response = ec2_client.request_spot_instances(
      InstanceCount=1,
      LaunchSpecification={
          'SecurityGroupIds': [
              GroupId,
          ],
          'SecurityGroups': [
              sg_name,
          ],
          'BlockDeviceMappings': [
              {
                  'DeviceName': '/dev/sda1',
                  'Ebs': {
                      'DeleteOnTermination': True,
                      'VolumeSize': ebs_volume,
                      'VolumeType': 'gp2',
                  },
              },
          ],
          'ImageId': 'ami-03f4878755434977f',
          'InstanceType': instance_type,
          'KeyName': keypair_name,
      },
    )

    # use a waiter on the instance to wait until running
    waiter = ec2_client.get_waiter('spot_instance_request_fulfilled')
    SpotRequestId = response["SpotInstanceRequests"][0]["SpotInstanceRequestId"]
    waiter.wait(SpotInstanceRequestIds=[
        SpotRequestId,
    ])
    spot_ec2_info = ec2_client.describe_spot_instance_requests(SpotInstanceRequestIds=[SpotRequestId])
    instance_id = spot_ec2_info["SpotInstanceRequests"][0]["InstanceId"]

    waiter = ec2_client.get_waiter('instance_running')
    waiter.wait(InstanceIds=[
        instance_id,
    ])

    ec2_info = ec2_client.describe_instances(InstanceIds=[
        instance_id,
    ])["Reservations"][0]["Instances"][0]
    print(ec2_info)
    ip = ec2_info["PrivateIpAddress"]
    public_ip = ec2_info["PublicIpAddress"]
    public_dns = ec2_info["PublicDnsName"]

  else:
    instance = ec2_resource.create_instances(
        BlockDeviceMappings=[
            {
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'DeleteOnTermination': True,
                    'VolumeSize': ebs_volume,
                    'VolumeType': 'gp2',
                },
            },
        ],
        ImageId='ami-03f4878755434977f', #ubuntu
        InstanceType=instance_type,
        KeyName=keypair_name,
        SecurityGroupIds=[
            GroupId,
        ],
        SecurityGroups=[
            sg_name
        ],
        MaxCount=1,
        MinCount=1,
    )

    # use a waiter on the instance to wait until running
    instance[0].wait_until_running()

    instance[0].reload()
    ip = instance[0].private_ip_address
    instance_id = instance[0].id
    public_ip = instance[0].public_ip_address
    public_dns = instance[0].public_dns_name

  # Rename the instance before leaving
  ec2_client.create_tags(Resources=[instance_id], Tags=[
      {
          'Key': 'Name',
          'Value': instance_name,
      },
  ])

  return instance_id, public_ip, public_dns, pkey_path


def send_cmd_pem(public_ip, username, pkey_path, command_list):

  key = paramiko.RSAKey.from_private_key_file(pkey_path)
  client = paramiko.SSHClient()
  client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

  # Connect/ssh to an instance
  try:
      # Here 'ubuntu' is user name and 'instance_ip' is public IP of EC2
      client.connect(hostname=public_ip, username=username, pkey=key)

      stdouts = []
      for command in command_list:
        print(f"Executing {command}")
        # Execute a command(cmd) after connecting/ssh to an instance
        stdin, stdout, stderr = client.exec_command(f"{command}")
        sout = stdout.read().decode('utf-8')
        print(sout)
        stdouts.append(sout)

      # close the client connection once the job is done
      client.close()

      return stdout

  except Exception as e:
      return e
  
def terminate_instance(access_key, secret_access_key, instance_id_to_terminate):
  
  # Define ec2 client and resource
  ec2_client = boto3.client('ec2', region_name=region_name, aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)
  ec2_resource = boto3.resource('ec2', region_name=region_name, aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)

  instances = ec2_resource.instances.filter(InstanceIds=[instance_id_to_terminate])
  for instance in instances:
    # Get only first security group
    group_name, group_id = instance.security_groups[0]['GroupName'], instance.security_groups[0]['GroupId']
    # Get the associated key_pair
    key_pair_name = instance.key_pair.name
    # Delete the instance and wait tillit is terminated
    instance = ec2_client.terminate_instances(InstanceIds=[instance_id_to_terminate])
    waiter = ec2_client.get_waiter('instance_terminated')
    print(f"Waiting for instance {instance_id_to_terminate} to be terminated...")
    waiter.wait(InstanceIds=[instance_id_to_terminate])
    # Delete the key-pair and security group
    response = ec2_client.delete_key_pair(
        KeyName=key_pair_name,
    )
    response = ec2_client.delete_security_group(
        GroupId=group_id,
        GroupName=group_name,
    )

  print("Instance and associated resources deleted permanently")

if __name__ == "__main__":

  pass
