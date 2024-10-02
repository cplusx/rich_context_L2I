export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
NNODE=$3
torchrun \
    --nnodes=$NNODE \
    --nproc_per_node 8 \
    --rdzv_id dist_train \
    --rdzv_backend c10d \
    --rdzv_endpoint $1:29500 \
    train.py -c $2 -n $NNODE -r

# usage: bash scripts/train_scripts/dist_train.sh 172.31.42.68 config_path.yaml num_nodes