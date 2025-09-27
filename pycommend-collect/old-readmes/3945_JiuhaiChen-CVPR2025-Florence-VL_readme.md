#  Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion.

[[Paper](https://arxiv.org/abs/2412.04424)] [[Project Page](https://jiuhaichen.github.io/florence-vl.github.io/)] [[Demo 8B](https://huggingface.co/spaces/jiuhai/Florence-VL-8B)]  [[Checkpoint 8B](https://huggingface.co/jiuhai/florence-vl-8b-sft)]


![Florence-VL Results](playground/data/results.png)


## News
* **[2024-12-05]** We release Arxiv paper, training code, checkpoint and Demo [[3B](https://huggingface.co/spaces/jiuhai/Florence-VL-3B), [8B](https://huggingface.co/spaces/jiuhai/Florence-VL-8B)]. 🤗 Have fun!


## Install Environment

1. Install package for tranining
```Shell
conda create -n florence-vl python=3.11 -y
conda activate florence-vl
pip install --upgrade pip  
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

2. Install package for evaluation (We use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for evaluation.)
```
cd lmms-eval
pip install -e .
```


## Dataset Download

1. Pretrain Data:

   Detailed Caption from [PixelProse](https://huggingface.co/datasets/tomg-group-umd/pixelprose) and [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V).

2. Instruction Data:

   TODO.

## Training Script
### Training with llama 3.1-8B (phi-3 is similar)

Set up your basic slurm information in the  ```scripts/florence-vl/llama/llama3.sh```
Then you can run pretrain and finetune job:


In ```scripts/florence-vl/llama/pretrain_llama.sh```, you need to manully export the following variable:

```shell
export NNODES=number of nodes

export DATA_PATH=/your/path/for/pretrain/data/json/file
export IMG=/your/image/folder

export OUTPUT=/checkpoint/save/path
```




In ```scripts/florence-vl/llama/finetune_llama.sh```, you need to manully export the following variable:


```shell
export NNODES=number of nodes

export DATA_PATH=/your/path/for/instuction/data/json/file
export IMG=/your/image/folder

export CKPT_PATH=/pretrain/checkpoint
export VIT_PATH=/pretrain/checkpoint/vision_tower
export OUTPUT=/checkpoint/save/path
```


## Evaluation Script

We use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for evaluation.

```shell
export OPENAI_API_KEY=your key
```

```shell
python -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/your/model/path/,conv_template=/choose/from/llama3/or/phi" \
    --tasks  textvqa_val,gqa,realworldqa,vizwiz_vqa_val,pope,scienceqa_img,mmvet,mme,seedbench,hallusion_bench_image,llava_in_the_wild,mathvista_testmini,docvqa_val,ocrbench,chartqa,ai2d,mmmu_val,mmbench_en_dev,infovqa_val,mmbench_cn_dev,mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix florence-vl \
    --output_path ./logs/
```



## Checkpoint 

1. Florence-VL 8B: [Pretrained Checkpoint](https://huggingface.co/jiuhai/florence-vl-8b-pretrain) and [Instructed Checkpoint](https://huggingface.co/jiuhai/florence-vl-8b-sft).
2. Florence-VL 3B: [Pretrained Checkpoint](https://huggingface.co/jiuhai/florence-vl-3b-pretrain) and [Instructed Checkpoint](https://huggingface.co/jiuhai/florence-vl-8b-sft).



## Acknowledgement

[LLaVA](https://github.com/haotian-liu/LLaVA): We start from codebase from the amazing LLaVA.

[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): Thanks for amazing multimodal evaluation codebase.




