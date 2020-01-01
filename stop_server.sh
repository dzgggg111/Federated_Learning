#!/bin/bash

ip=$1
key_file=$2
ssh  -i ${key_file} -o "StrictHostKeyChecking=no" ubuntu@$ip "sh -c 'pkill python*'"
