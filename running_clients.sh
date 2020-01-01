#!/bin/bash


ip=$1
client_number=$2
client_tier=$3
poisoning=$4
key_file=$5
scp -i ${key_file} -o "StrictHostKeyChecking=no" -r Federated_Learning ubuntu@$ip:/home/ubuntu/
ssh  -i ${key_file} -o "StrictHostKeyChecking=no" ubuntu@$ip "sh -c 'cd /home/ubuntu/Federated_Learning/; python3 -u fl_client.py $client_number $client_tier $poisoning > experiment.log 2>&1 &'"
