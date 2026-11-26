#!/bin/bash
# Watches the 3 shootout VMs; pulls results when each finishes; terminates all at the end.
KEY=~/.ssh/alt-direction-probe.pem
OUT="/Users/varundaiya/CI-1/Alternate direction/results/shootout"
mkdir -p "$OUT"
declare -A DONE
VMS=("i-0995c14348add3b8b" "i-0e981185163217e86" "i-08b9ca8d2b18aef30")
KINDS=("spark" "mcp2b" "mcp1b")

pull_vm () {
  local ID=$1 KIND=$2
  local STATE
  STATE=$(aws ec2 describe-instances --region us-east-1 --instance-ids $ID --query 'Reservations[0].Instances[0].State.Name' --output text)
  if [ "$STATE" = "stopped" ]; then
    aws ec2 start-instances --region us-east-1 --instance-ids $ID >/dev/null
    sleep 45
  fi
  local IP
  IP=$(aws ec2 describe-instances --region us-east-1 --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  for i in $(seq 1 20); do ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes ubuntu@$IP 'echo OK' 2>/dev/null && break; sleep 10; done
  if [ "$KIND" = "spark" ]; then
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/shootout/results/fastbench_spark.json" "$OUT/" 2>/dev/null
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/shootout/results/throughput_spark.json" "$OUT/" 2>/dev/null
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/shootout/results/aime_aime2"*.json "$OUT/" 2>/dev/null
    scp -i $KEY -o BatchMode=yes -r "ubuntu@$IP:~/shootout/results/lmeval_spark_gsm8k" "$OUT/" 2>/dev/null
  else
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/bench/results/fastbench_${KIND}.json" "$OUT/" 2>/dev/null
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/bench/results/throughput_${KIND}.json" "$OUT/" 2>/dev/null
  fi
  aws ec2 terminate-instances --region us-east-1 --instance-ids $ID >/dev/null
  echo "$KIND collected+terminated"
}

while true; do
  ALL=1
  for idx in 0 1 2; do
    ID=${VMS[$idx]}; KIND=${KINDS[$idx]}
    [ "${DONE[$KIND]:-0}" = "1" ] && continue
    STATE=$(aws ec2 describe-instances --region us-east-1 --instance-ids $ID --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
    if [ "$STATE" = "stopped" ]; then
      pull_vm $ID $KIND && DONE[$KIND]=1
    elif [ "$STATE" = "running" ]; then
      IP=$(aws ec2 describe-instances --region us-east-1 --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
      FLAG="DONE"; [ "$KIND" = "spark" ] && FLAG="SPARK_FAST_DONE"
      DDIR="bench"; [ "$KIND" = "spark" ] && DDIR="shootout"
      if ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes ubuntu@$IP "test -f ~/$DDIR/results/$FLAG" 2>/dev/null; then
        pull_vm $ID $KIND && DONE[$KIND]=1
      fi
    fi
    [ "${DONE[$KIND]:-0}" != "1" ] && ALL=0
  done
  [ "$ALL" = "1" ] && { echo "ALL COLLECTED"; exit 0; }
  sleep 240
done