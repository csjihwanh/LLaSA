echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
export PYTHONPATH=$PYTHONPATH:/workspace/LLaSA/
export CUDA_VISIBLE_DEVICES=0

echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=1
NUM_CPU=128

:'
torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \'

python utils/train.py > utils/train_log.txt 2>&1 &