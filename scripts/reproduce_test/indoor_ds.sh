#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/vissar_trainval.py"
main_cfg_path="configs/loftr/outdoor/loftr_ds_dense.py"
ckpt_path="weights/multimodal.ckpt" 
dump_dir="dump/per_tps"   
profiler_name="inference"
n_nodes=1  
n_gpus_per_node=1      
torch_num_workers=4
batch_size=1  

python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes}  \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark 
    