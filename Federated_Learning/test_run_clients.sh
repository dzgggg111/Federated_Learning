#!/bin/bash

ec2_multicpu(){
	LOG_FOLDER="./logs"
	rm -rf ${LOG_FOLDER}; mkdir ${LOG_FOLDER}
	echo "Starting Clients..."
	clients50_tier_index_groups=("1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "2" "2" "2" "2" "2" "2" "2" "2" "2" "2" "3" "3" "3" "3" "3" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5" "5" "5" "5" "5" "5")
	clients10_tier_index_groups=("1" "1" "2" "2" "3" "3" "4" "4" "5" "5")
	clients25_tier_index_groups=("1" "1" "1" "1" "1" "2" "2" "2" "2" "2" "3" "3" "3" "3" "3" "4" "4" "4" "4" "4" "5" "5" "5" "5" "5")
	for test in {0..49} ; do
		echo 'Client ' $test ' is starting...'
		TIER=${clients50_tier_index_groups[test]}
		python3 -u fl_client.py ${test} ${TIER} &>> ./${LOG_FOLDER}/myclient_${test}.log &
		sleep 2
	done
}

ec2_multicpu
