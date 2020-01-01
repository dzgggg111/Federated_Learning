#!/bin/bash


echo deploying clients

#id=$(aws ec2 run-instances --image-id ami-0e57276a34295308f --count 1 --instance-type c5.large  --security-group-ids sg-0e1d9d0d6b6f90fc5 --subnet-id subnet-8b60bcc1 | grep InstanceId)

#echo $id 
aws ec2 run-instances --image-id ami-0e57276a34295308f --count 1 --instance-type c5.large  --security-group-ids sg-0e1d9d0d6b6f90fc5 --subnet-id subnet-8b60bcc1
