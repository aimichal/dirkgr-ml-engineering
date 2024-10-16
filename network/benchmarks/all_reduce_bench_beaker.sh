#!/usr/bin/env bash

set -exuo pipefail

NUM_NODES=2

if [ -z "$BEAKER_JOB_ID" ]; then
  gantry run \
    --workspace ai2/13B \
    --task-name all_reduce_bench \
    --description "Benchmarking all_reduce performance" \
    --priority normal \
    --preemptible \
    --beaker-image petew/olmo-torch23-gantry \
    --cluster ai2/jupiter-cirrascale-2 \
    --gpus 8 \
    --replicas "${NUM_NODES}" \
    --leader-selection \
    --host-networking \
    --budget ai2/oe-training \
    --no-nfs \
    --propagate-failure \
    --propagate-preemption \
    --synchronized-start-timeout 5m \
    --no-python \
    --env OMP_NUM_THREADS=16 \
    --shared-memory 10GiB \
    --yes \
    --timeout=-1 \
    -- /bin/bash ml-engineering/network/benchmarks/all_reduce_bench_beaker.sh
else
  conda shell.bash activate base
  export NCCL_DEBUG=INFO
  export NCCL_IB_HCA="^=mlx5_bond_0"
  export NCCL_SOCKET_IFNAME=ib
  export TORCH_DIST_INIT_BARRIER=1
  torchrun \
    --nnodes "${NUM_NODES}:${NUM_NODES}" \
    --nproc-per-node 8 \
    --rdzv_id 12347 \
    --rdzv_backend static \
    --rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" \
    --node_rank "${BEAKER_REPLICA_RANK}" \
    --rdzv_conf 'read_timeout=420' \
    ml-engineering/network/benchmarks/all_reduce_bench.py
fi
