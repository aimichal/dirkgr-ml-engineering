#!/usr/bin/env bash

set -exuo pipefail

NUM_NODES=2

set -u

conda shell.bash activate base
export NCCL_DEBUG=INFO
export NCCL_IB_HCA="^=mlx5_bond_0"
export NCCL_SOCKET_IFNAME=ib
export TORCH_DIST_INIT_BARRIER=1

curl -O https://raw.githubusercontent.com/dirkgr/ml-engineering/refs/heads/Beaker/network/benchmarks/all_reduce_bench.py

torchrun \
	--nnodes "${NUM_NODES}:${NUM_NODES}" \
	--nproc-per-node 8 \
	--rdzv_id 12347 \
	--rdzv_backend static \
	--rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" \
	--node_rank "${BEAKER_REPLICA_RANK}" \
	--rdzv_conf 'read_timeout=420' \
	all_reduce_bench.py
