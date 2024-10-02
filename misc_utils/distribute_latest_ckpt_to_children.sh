EXPT_NAME=$1
shift # Skip the first argument as we have already used it

CHILDREN_NODES=("$@") # Assign all remaining arguments to the array

for children in "${CHILDREN_NODES[@]}"; do
    echo "Copying to ${children}"
    scp -o StrictHostKeyChecking=no "experiments/${EXPT_NAME}/last.ckpt/latest" "ubuntu@${children}:/home/ubuntu/rich_context_layout2image/experiments/${EXPT_NAME}/last.ckpt/"
    scp -o StrictHostKeyChecking=no "experiments/${EXPT_NAME}/last.ckpt/checkpoint/mp_rank_00_model_states.pt" "ubuntu@${children}:/home/ubuntu/rich_context_layout2image/experiments/${EXPT_NAME}/last.ckpt/checkpoint/"
done

# EXPT_NAME=$1
# CHILDREN_NODES=(
#     "172.31.47.170" # 4
#     "172.31.36.253" # 3
#     "172.31.47.194" # 1
# )

# for children in "${CHILDREN_NODES[@]}"; do
#     scp -o StrictHostKeyChecking=no "experiments/${EXPT_NAME}/last.ckpt/latest" "ubuntu@${children}:/home/ubuntu/rich_context_layout2image/experiments/${EXPT_NAME}/last.ckpt/"
#     scp -o StrictHostKeyChecking=no "experiments/${EXPT_NAME}/last.ckpt/checkpoint/mp_rank_00_model_states.pt" "ubuntu@${children}:/home/ubuntu/rich_context_layout2image/experiments/${EXPT_NAME}/last.ckpt/checkpoint/"
# done