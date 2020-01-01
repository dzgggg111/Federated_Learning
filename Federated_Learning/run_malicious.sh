#!/bin/bash


LOGS_DIR=./logs
if [ -d "${LOGS_DIR}" ]; then
        echo "Logs dir already exists, deleting it."
        rm -rf ${LOGS_DIR}
fi
mkdir ${LOGS_DIR}


TIER=1
		client_num = 0
		
        actual_client_num=$((client_num))
		
        echo "Deploying Client ${actual_client_num}"
		
        python3 fl_client.py ${actual_client_num} ${TIER} "False" &>> ./logs/client_${actual_client_num}.log &
		
        sleep 2
		
        echo "Deployed"

		
        actual_client_num=1
		
        echo "Deploying Client ${actual_client_num}"
		
        python3 fl_client.py ${actual_client_num} ${TIER} "False" &>> ./logs/client_${actual_client_num}.log &
		
        sleep 2
		
        echo "Deployed"
		
        actual_client_num=2
		
        echo "Deploying Client ${actual_client_num}"
		
        python3 fl_client.py ${actual_client_num} ${TIER} "True" &>> ./logs/client_${actual_client_num}.log &
		
        sleep 2
		
        echo "Deployed"
		
        actual_client_num=3
		
        echo "Deploying Client ${actual_client_num}"
		
        python3 fl_client.py ${actual_client_num} ${TIER} "True" &>> ./logs/client_${actual_client_num}.log &
		
        sleep 2
		
        echo "Deployed"
		
        actual_client_num=4
		
        echo "Deploying Client ${actual_client_num}"
		
        python3 fl_client.py ${actual_client_num} ${TIER} "True" &>> ./logs/client_${actual_client_num}.log &
		
        sleep 2
		
        echo "Deployed"
		
echo "Deployed all clients"
