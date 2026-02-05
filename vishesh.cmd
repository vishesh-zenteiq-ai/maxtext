1. 
conda create -n maxtext-cuda13 python=3.12 -y
conda activate maxtext-cuda13

2.
python -m pip install -U "jax[cuda13]"

v//// can verify
python -c "import jax; print(jax.devices())"

3.
python -m pip install uv

# Editable install with CUDA 13 extra and pinned deps:
UV_HTTP_TIMEOUT=600 uv pip install -e ".[cuda13]" --resolution=lowest
install_maxtext_github_deps  # if you use the extras script

cfg/////
python -m pip install nvidia-cudnn-cu13 nvidia-nccl-cu13

# Only if headers aren’t in standard CUDA include dirs on that machine:
export CUDNN_INCLUDE="$(python -c 'import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), \"include\"))')"
export NCCL_INCLUDE="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/include"
export NVTX_INCLUDE="/usr/local/cuda/include"  # or your nvtx3 include dir

export CFLAGS="-I$CUDNN_INCLUDE -I$NCCL_INCLUDE -I$NVTX_INCLUDE ${CFLAGS:-}"
export CPPFLAGS="-I$CUDNN_INCLUDE -I$NCCL_INCLUDE -I$NVTX_INCLUDE ${CPPFLAGS:-}"
UV_HTTP_TIMEOUT=600 uv pip install -e ".[cuda13]" --resolution=lowest


4.
run////
cd ~/github/maxtext  # repo root
python -c "import MaxText; print(MaxText.__version__)"
python -c "import jax; print(jax.devices())"  # confirm GPUs

python -m MaxText.train \
  src/MaxText/configs/base.yml \
  hardware=gpu \
  dataset_type=synthetic \
  steps=1 \
  base_output_directory=/tmp/maxtext_cuda13_test \
  enable_checkpointing=false





import error , this is just a sample train cmd:
python3 -m MaxText.train MaxText/configs/base.yml     run_name=llama_tinystories_run     base_output_directory=/home/parani/arnav/output
     model_name=qwen3-0.6b     dataset_type=hf     hf_path=roneneldan/TinyStories     train_data_columns="['text']"     hardware=gpu     attention=dot_product     tokenizer_path="/home/parani/arnav/Qwen" 
    tokenizer_type=huggingface                                                                                                                                                                              │
                                                                       