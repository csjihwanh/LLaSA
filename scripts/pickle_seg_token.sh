echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
export CUDA_VISIBLE_DEVICES=0

echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=1
NUM_CPU=128

torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    model/segmentation/token_pickler.py