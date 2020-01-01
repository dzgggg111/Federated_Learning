#!/bin/bash

ip=$1
file_name=$2
key_file=$3
scp -i ${key_file} -o "StrictHostKeyChecking=no" -r Federated_Learning ubuntu@$ip:/home/ubuntu/

ssh  -i ${key_file} -o "StrictHostKeyChecking=no" ubuntu@$ip "sh -c 'cd /home/ubuntu/Federated_Learning/; python3 -u fl_server.py > $file_name 2>&1 &'"

