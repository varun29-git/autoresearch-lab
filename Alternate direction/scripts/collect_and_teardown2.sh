#!/bin/bash
# bash-3.2-safe collector: waits for all 3 VMs to self-stop, then start->pull->terminate each.
KEY=~/.ssh/alt-direction-probe.pem
OUT="/Users/varundaiya/CI-1/Alternate direction/results/shootout"
mkdir -p "$OUT"
VMS="i-0995c14348add3b8b i-08b9ca8d2b18aef30 i-0db1f07e156edfb2e"
KINDS="spark mcp1b mcp2b"

pull_one () {
  ID=$1; KIND=$2
  aws ec2 start-instances --region us-east-1 --instance-ids $ID >/dev/null 2>&1
  sleep 45
  IP=$(aws ec2 describe-instances --region us-east-1 --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  for i in $(seq 1 20); do ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes ubuntu@$IP 'echo OK' 2>/dev/null && break; sleep 10; done
  if [ "$KIND" = "spark" ]; then
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/shootout/results/fastbench_spark.json" "$OUT/" 2>/dev/null
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/shootout/results/throughput_spark.json" "$OUT/" 2>/dev/null
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/shootout/results/aime_aime25_spark.json" "$OUT/" 2>/dev/null
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/shootout/results/aime_aime26_spark.json" "$OUT/" 2>/dev/null
    scp -i $KEY -o BatchMode=yes -r "ubuntu@$IP:~/shootout/results/lmeval_spark_gsm8k" "$OUT/" 2>/dev/null
  else
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/bench/results/fastbench_${KIND}.json" "$OUT/" 2>/dev/null
    scp -i $KEY -o BatchMode=yes "ubuntu@$IP:~/bench/results/throughput_${KIND}.json" "$OUT/" 2>/dev/null
  fi
  aws ec2 terminate-instances --region us-east-1 --instance-ids $ID >/dev/null
  echo "$KIND collected+terminated"
}

while true; do
  REMAIN=0
  for ID in $VMS; do
    STATE=$(aws ec2 describe-instances --region us-east-1 --instance-ids $ID --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
    [ "$STATE" = "running" ] || [ "$STATE" = "pending" ] && REMAIN=1
  done
  if [ "$REMAIN" = "0" ]; then
    set -- $KINDS
    for ID in $VMS; do pull_one $ID $1; shift; done
    echo "ALL COLLECTED"
    exit 0
  fi
  sleep 240
done