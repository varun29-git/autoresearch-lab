# Infra record — Spark-X2.6-1.7B feasibility probes (2026-11-12)

- VM: AWS EC2 `g5.xlarge` (1× NVIDIA A10G 24 GB, 4 vCPU, 16 GB RAM), region **us-east-1**
  (ap-south-1 had zero G-family vCPU quota on this account).
- Instance ID: `i-0a318f89caf19649e`, name tag `alt-direction-probe`.
- AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20261103-era build
  (`ami-020971c2de8d4cbca` in us-east-1), driver 595.112.07, CUDA 13.2, torch 2.6.0+cu124.
- Key pair: `alt-direction-probe` (private key at `~/.ssh/alt-direction-probe.pem`, NOT in repo).
- Security group: `alt-direction-probe-sg` (`sg-05ef13d06a16a4c2b`), SSH from 103.246.194.178/32 only.
- Root volume: 150 GB gp3.
- On-demand price: ~$1.006/hr (g5.xlarge us-east-1).
- Mumbai G-family quota was 0; us-east-1 quota 64 vCPUs. If Mumbai GPUs are wanted later,
  a service-quota increase request is needed.

## Dataset note
FLORES-200 canonical sources (GitHub raw, dl.fbaipublicfiles.com, facebook/flores on HF
which is gated) were all unreachable on 2026-11-12. Probes use the **wikimedia/wikipedia
20231101 dump** (streaming, first ~1200 chars of the first 200–250 articles per language).
Caveat: Wikipedia is almost certainly inside the pretraining data of all models probed, so
absolute perplexities are optimistic; cross-language and cross-model comparisons remain valid.

## VM #2 — benchmark shootout (2026-11-12 evening)
- Instance `i-0995c14348add3b8b` ("minicpm-shootout"), g5.xlarge A10G, us-east-1a,
  **on-demand** (~$1.006/hr) — spot g5/g6 had zero capacity in us-east-1 all AZs.
  EBS root 200GB gp3 with DeleteOnTermination=false (autosave-by-EBS + results
  scp'd back after every benchmark to `Alternate direction/results/shootout/`).
- IP: 100.53.109.164. Workload: Spark-X2.6-1.7B vs MiniCPM5-1B/2B on
  GSM8K/AIME25-26/GPQA-D/IFEval/HumanEval/IFBench/BFCL + throughput.

## Laptop-off mode (armed 2026-11-12 ~16:45 UTC)
Both VMs are self-driving and self-stopping (no laptop dependency):
- VM #1 (i-0a318f89caf19649e): runs GSM8K-Indic then fill-run, then `shutdown -h now`
  (armed watcher on fill_done.flag). Results stay on its EBS root.
- VM #2 (i-0995c14348add3b8b): `~/shootout/master.sh` runs the full benchmark pipeline
  (run_model.sh × Spark, MiniCPM5-1B, MiniCPM5-2B), touches results/SHOOTOUT_DONE,
  then `shutdown -h now`. Results in ~/shootout/results on the 200GB volume
  (vol-01368525bd8b9c8e5, tagged minicpm-shootout).

## Parallelized shootout (2026-11-13 ~05:30 UTC, after overnight stall)
- VM #2 i-0995c14348add3b8b (100.53.109.164): spark_finish.sh = gpqa(4096) → ifeval →
  humaneval → throughput → touch results/SPARK_DONE. Stays RUNNING (pull + terminate manually).
- VM #3 i-0e981185163217e86 (174.129.170.148, "mcp2b-bench"): MiniCPM5-2B full bench
  (gsm8k → aime25 → gpqa → ifeval → humaneval → throughput), self-shutdown via ~/bench/mcp_vm_setup.sh.
- VM #4 i-08b9ca8d2b18aef30 (54.204.162.107, "mcp1b-bench"): MiniCPM5-1B, same, self-shutdown.
- VM #1 i-0a318f89caf19649e: TERMINATED 2026-11-13 morning (all Indic results pulled).
- Settings parity: all models temp 1.0 / top_p 0.95 / thinking high; GPQA max-new 4096 all
  (Spark rerun at 4096 for parity); AIME25 4 samples × max-new 8192 all. AIME26 = Spark only.
- Completed Spark numbers so far: GSM8K 77.7 flex, AIME25 6.7%, AIME26 0.83%.

## On session resume (laptop back on)
1. `aws ec2 describe-instances --region us-east-1 --instance-ids i-0a318f89caf19649e i-0995c14348add3b8b --query 'Reservations[*].Instances[*].[InstanceId,State.Name]'`
2. Start stopped instances (`aws ec2 start-instances`), wait for SSH
3. scp results: VM1 `~/probe/results/` (incl. evals/) → `Alternate direction/results/`;
   VM2 `~/shootout/results/` → `Alternate direction/results/shootout/`
4. Terminate both instances; delete vol-01368525bd8b9c8e5 if it lingers
5. Delete SG sg-05ef13d06a16a4c2b + key pairs alt-direction-probe (us-east-1, ap-south-1)
   and ap-south-1 SG sg-0c3f598bf88a9b570
6. Write the final consolidated report.
