from __future__ import print_function
import boto3
from time import sleep
import subprocess
import os, sys, threading

# Config ------------------------------------------------
AVAIL_INSTANCE_TYPES = { 0: 'c5.2xlarge', 1 : 'c5.xlarge', 2 : 'c5.large' } # Tier Num: Instance Type
AMI_ID = 'ami-01707119cd46c7980' # Client Image Instance Id
PEM_KEY_FILE = 'ec2_aws_final' # pem filename, must be in the same directory and chmodded to 400
SECURITY_GROUP_IDS = ['sg-03dedf55577aed8c5']
SERVER_IDS = ['i-07538c110b2df8643']

def _start_client(ip, index, tier, poisoning):
    key_file = PEM_KEY_FILE + '.pem'
    myarg = './running_clients.sh' + ' ' + ip + ' ' + str(index) + ' ' + str(tier) + ' ' + poisoning + ' ' + key_file
    subprocess.call(myarg, shell=True)
    return 

def start_instances(tier_num, client_index_offset, num_instances=10, poison='False'):
    '''Deploy instances with specified hardware, run fl_clients on instances.'''
    ec2 = boto3.client('ec2')
    instance_type = AVAIL_INSTANCE_TYPES[tier_num]
    response = ec2.run_instances(
                   ImageId=AMI_ID,
                   MinCount=num_instances, 
                   MaxCount=num_instances, 
                   InstanceType=instance_type,
                   KeyName=PEM_KEY_FILE,
                   SecurityGroupIds=SECURITY_GROUP_IDS)
    ips, ids = [], []
    for instance in response['Instances']:
        ids.append(instance['InstanceId'])
    with open('instance_list.list', 'a+') as fp:
        for id in ids:
            fp.write(id + '\n')
    print('Waiting for all clients to be ready. Sleeping....')
    sleep(60)
    print('Start deploying clients in instances...')
    response = ec2.describe_instances(InstanceIds=ids)
    thread_pool = []
    for index, instance in enumerate(response['Reservations'][0]['Instances']):
        i = index + client_index_offset
        ip = instance['PublicIpAddress']
        print('Starting Client thread\n\tIP:{}\n\tINDEX:{}\n\tTIER:{}\n\tPOISON:{}'.format(ip, i, tier_num, poison))
        t = threading.Thread(target=_start_client, args=(ip, i, tier_num, poison))
        t.start()
        thread_pool.append(t)
    for t in thread_pool:
        t.join()
    print('Finished deploying clients instances.')
    return

def start_server():
    key_file = PEM_KEY_FILE + '.pem'
    ec2 = boto3.client('ec2')
    ids = SERVER_IDS
    response = ec2.describe_instances(InstanceIds=ids)
    ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
    myarg = './deploy_server.sh' + ' ' + ip + ' ' + 'server.log' + ' ' + key_file
    subprocess.call(myarg, shell=True)

def terminate_instances():
    ec2 = boto3.client('ec2')
    key_file = PEM_KEY_FILE + '.pem'
    ids = []
    server_ids = SERVER_IDS
    response = ec2.describe_instances(InstanceIds=server_ids)
    ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
    myarg = 'scp -i "{}" -o "StrictHostKeyChecking=no" ubuntu@{}:/home/ubuntu/Federated_Learning/server.log ./'.format(key_file, ip)
    subprocess.call(myarg, shell=True)
    myarg = 'scp -i "{}" -o "StrictHostKeyChecking=no" ubuntu@{}:/home/ubuntu/Federated_Learning/train_valid_old.csv ./'.format(key_file, ip)
    subprocess.call(myarg, shell=True)
    myarg = 'scp -i "{}" -o "StrictHostKeyChecking=no" ubuntu@{}:/home/ubuntu/Federated_Learning/test_old.csv ./'.format(key_file, ip)
    subprocess.call(myarg, shell=True)
    myarg = './stop_server.sh' + ' ' + ip + ' ' + key_file
    subprocess.call(myarg, shell=True)
    with open('instance_list.list', 'r') as fp:
        for line in fp.readlines():
            ids.append(str(line).rstrip('\n'))
    ec2.terminate_instances(InstanceIds=ids)
    if os.path.exists("instance_list.list"):
        os.remove('instance_list.list')



if __name__ == '__main__':
    
    if sys.argv[1] == 'start':
        if os.path.exists("instance_list.list"):
            os.remove("instance_list.list")
        start_instances(tier_num=1, client_index_offset=0, num_instances=2, poison='True')
        start_instances(tier_num=1, client_index_offset=2, num_instances=8, poison='False')
        start_instances(tier_num=2, client_index_offset=10, num_instances=2, poison='True')
        start_instances(tier_num=2, client_index_offset=12, num_instances=8, poison='False')
        start_server()
    elif sys.argv[1] == 'stop':
        terminate_instances()    
    else:
        print('Not valid args')
