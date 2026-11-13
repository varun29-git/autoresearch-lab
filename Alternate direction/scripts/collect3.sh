#!/bin/bash
# Collector v3: waits for the 3 fastbench VMs to self-stop, then start->pull->terminate.
KEY=~/.ssh/alt-direction-probe.pem
OUT="/Users/varundaiya/CI-1/Alternate direction/results/shootout"
mkdir -p "$OUT"
VMS="i-01881a865fe42d498 i-0172faa929ab6ceff i-04a16041830edb6ca"
KINDS="spark mcp2b mcp1b"

pull_one () {
  ID=$1; KIND=$2
  aws ec2 start-instances --region us-east-1 --instance-ids $ID >/dev/null 2>&1
  sleep 45
  IP=$(aws ec2 describe-instances --region us-east-1 --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  for i in $(seq 1 20); do ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes ubuntu@$IP 'echo OK' 2>/dev/null && break; sleep 10; done
  scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/bench/results/fastbench_${KIND}.json" "$OUT/" 2>/dev/null
  scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/bench/logs/fastbench_${KIND}.log" "$OUT/${KIND}_fastbench.log" 2>/dev/null
  aws ec2 terminate-instances --region us-east-1 --instance-ids $ID >/dev/null
  echo "$KIND collected+terminated"
}

while true; do
  REMAIN=0
  for ID in $VMS; do
    STATE=$(aws ec2 describe-instances --region us-east-1 --instance-ids $ID --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
    if [ "$STATE" = "running" ] || [ "$STATE" = "pending" ]; then REMAIN=1; fi
  done
  if [ "$REMAIN" = "0" ]; then
    set -- $KINDS
    for ID in $VMS; do pull_one $ID $1; shift; done
    echo "ALL COLLECTED"
    exit 0
  fi
  sleep 240
done