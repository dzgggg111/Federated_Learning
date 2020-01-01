#!/bin/bash


LOGS_DIR=./logs
if [ -d "${LOGS_DIR}" ]; then
        echo "Logs dir already exists, deleting it."
        rm -rf ${LOGS_DIR}
fi
mkdir ${LOGS_DIR}

client_index_offset=0
TIER=1
for client_num in {0..9}; do
        actual_client_num=$((client_num+client_index_offset))
        echo "Deploying Client ${actual_client_num}"
        python3 fl_client.py ${actual_client_num} ${TIER} "True" &>> ./logs/client_${actual_client_num}.log &
        sleep 2
        echo "Deployed"
done

client_index_offset=10
TIER=2
for client_num in {0..9}; do
        actual_client_num=$((client_num+client_index_offset))
        echo "Deploying Client ${actual_client_num}"
        python3 fl_client.py ${actual_client_num} ${TIER} "True" &>> ./logs/client_${actual_client_num}.log &
        sleep 2
        echo "Deployed"
done

echo "Deployed all clients"
