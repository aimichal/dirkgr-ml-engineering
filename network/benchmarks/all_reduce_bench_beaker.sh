#!/usr/bin/env bash

set -exuo pipefail

NUM_NODES=$1
shift

set -u

conda shell.bash activate base

# ------------------------------------------------
# Environment variables set by running this on an augusta machine:
#
#   NCCL_LIB_DIR=/var/lib/tcpxo/lib64 source /var/lib/tcpxo/lib64/nccl-env-profile.sh
#
# Some of the vars there are repeated (like NCCL_FASTRAK_IFNAME)
# so they are deduplicated below.
# ------------------------------------------------

export LD_LIBRARY_PATH="/var/lib/tcpxo/lib64:${LD_LIBRARY_PATH}"
export NCCL_CROSS_NIC=0
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=4
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_FASTRAK_NUM_FLOWS=2
export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_FASTRAK_USE_SNAP=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
export NCCL_TUNER_PLUGIN=libnccl-tuner.so
export NCCL_TUNER_CONFIG_PATH=/var/lib/tcpxo/lib64/a3plus_tuner_config.textproto
export NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/var/lib/tcpxo/lib64/a3plus_guest_config.textproto
export NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN # was WARN in nccl-env-profile.sh
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV,COLL,GRAPH
export NCCL_FASTRAK_CTRL_DEV=enp0s12
export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
export NCCL_SOCKET_IFNAME=enp0s12
#export NCCL_USE_SNAP=1
#export NCCL_FASTRAK_USE_LLCM=1
#export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices

# ------------------------------------------------
# End Augusta env vars
# ------------------------------------------------

export TORCH_DIST_INIT_BARRIER=1

curl -O https://raw.githubusercontent.com/aimichal/dirkgr-ml-engineering/refs/heads/Beaker/network/benchmarks/all_reduce_bench.py

torchrun \
	--nnodes "${NUM_NODES}:${NUM_NODES}" \
	--nproc-per-node 8 \
	--rdzv_id 12347 \
	--rdzv_backend static \
	--rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" \
	--node_rank "${BEAKER_REPLICA_RANK}" \
	--rdzv_conf 'read_timeout=420' \
	all_reduce_bench.py
