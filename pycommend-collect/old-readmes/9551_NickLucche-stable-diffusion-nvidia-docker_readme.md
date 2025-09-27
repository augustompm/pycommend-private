A friend of mine working in art/design wanted to try out [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) on his own GPU-equipped PC, but he doesn't know much about coding, so I thought that baking a quick docker build was an easy way to help him out. This repo holds the files that go into that build.

I also took the liberty of throwing in a simple web UI (made with gradio) to wrap the model. Perhaps we can evolve it a bit to offer a few more functionalities (see TODO).

**UPDATE:** we now support inference on multiple GPUs with a "Data Parallel" approach.

~~**UPDATE 2:** we now support inference on multiple GPUs with a "Model Parallel" approach (see `Multi-GPU` section).~~

**UPDATE 3 but really it's a v2:** [Stable Diffusion 2.0](https://stability.ai/blog/stable-diffusion-v2-release) is out generating images more beautiful than ever! This is now the default model being loaded and it supports all previous features and more. I've also added support for *img2img* and *image inpainting* and refreshed the UI, give it a try! 

# Requirements
 - OS: Ubuntu (tested on 20.04) or Windows (tested on Windows 10 21H2)
 - Nvidia GPU with at least 6GB vRAM (gtx 700 onward, please refer [here](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)). Mind that the bigger the image size (or the number of images) you want to dream, the more memory you're gonna need. For reference, dreaming a 256x256 image should take up ~5gb, while a 512x512 around 7gb. 
 - Free Disk space > 2.8gb
 - Docker and Nvidia-docker.
 - HuggingFace account as well as ~~registration to this repository https://huggingface.co/CompVis/stable-diffusion-v1-4 (simply click on `Access Repository`)~~. No longer needed if you use default v2 model (see "About model versions" below).

# Installation

First of all, make sure to have docker and nvidia-docker installed in your machine.

**Windows users**: [install WSL/Ubuntu](https://stackoverflow.com/a/56783810) from store->install [docker](https://docs.docker.com/desktop/windows/wsl/) and start it->update Windows 10 to version 21H2 (Windows 11 should be ok as is)->test out [GPU-support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl2) (a simple `nvidia-smi` in WSL should do). If `nvidia-smi` does not work from WSL, make sure you have updated your nvidia drivers from the official app. 

The easiest way to try out the model is to simply use the pre-built image at `nicklucche/stable-diffusion`.   

My advice is that you start the container with:

`docker run --name stable-diffusion --pull=always --gpus all -it -p 7860:7860 nicklucche/stable-diffusion` 

the *first time* you run it, as it will download the model weights (can take a few minutes to do so) and store them on disk (as long as you don't delete the container).
Then you can simply do `docker stop stable-diffusion` to stop the container and `docker start stable-diffusion` to bring it back up whenever you need.
`--pull=always` is to make sure you get the latest image from dockerhub, you can skip it if you already have it locally.

Once the init phase is finished a message will pop-up in your terminal (`docker logs stable-diffusion`) and you should be able to head to http://localhost:7860/ in your favorite browser and see something like this:

![](assets/screen.png)

By default, the half-precision/fp16 model is loaded. This is the recommended approach if you're planning to run the model on a GPU with < 10GB of memory (takes half the space, ~half the time and yields similar output). To disable FP16 and run inference using single-precision (FP32), set the environment variable FP16=0 as a docker run option, like so:

`docker run .. -e FP16=0 ...`  

## Multi-GPU

The model can be run in both a "DataParallel" or a combined "Model+Data Parallel" fashion to speed up inference time and leverage your multi-gpu setup to its fullest.

### Data Parallel

This means that the model is replicated over multiple GPUs, each handled by a separate sub-process. By default, the model runs on device 0 (no parallelism). You can change that by specifying the desired device(s) by adding one of the following options:

 - `-e DEVICES=1 ...` runs model on GPU 1 (starts from 0)
 - `-e DEVICES=0,1 ...` runs model on GPU 0 and 1
 - `-e DEVICES=all ...` runs model on all available GPUs

Each device/model generates a full image, so make sure you increase the `Number of Images` slider to generate multiple images in parallel!
(Single image generation speed won't be affected).

I should also mention that adding the nsfw filter (by checking corresponding box) includes moving an additional model to GPU, so it can cause out of memory issues.

### ~~Model Parallel~~ -Currently disabled! Use "Data Parallel" for true parallelism!-

It works by splitting the model into a fixed number of parts, assigning each part to a device and then handling data transfer from one device to the other (more technical details [here](https://github.com/NickLucche/stable-diffusion-nvidia-docker/issues/8) or from source).
This was originally intended to support setups that had GPUs with small amounts of VRAM that could only run the model by combining their resources, but now it also supports splitting multiple models to accomodate for bigger GPUs, effectively combining Model and Data Parallel.

Single image inference will be slower in this modality (since we may need to move data from one device to the other), but it allows to fill your memory more efficiently if you have big GPUs by creating multiple models.
You can try out this option with:

`-e MODEL_PARALLEL=1` 

Note that if your system has highly imbalanced GPU memory distribution (e.g. gpu0->6Gb, gpu1->24Gb.. ) the smallest device might bottleneck the inference process; the easiest way to fix that, is to ignore the smallest device by *not* specifying it in the `DEVICES` list (e.g. `-e DEVICES=1,2..`).

## About models

By default, the model loaded is [stabilityai/stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base). Many other checkpoints have been created that are compatible with [diffusers](https://github.com/huggingface/diffusers) (awesome library, ckeck it out) and you can provide them as an additional environment variable like so:

`-e MODEL_ID=runwayml/stable-diffusion-v1-5`

Model weights are downloaded to and loaded from `/root/.cache/huggingface/diffusers`, so if you want to share your model across multiple containers runs, you can provide this path as a [docker volume](https://docs.docker.com/storage/volumes/):

`-v /path/to/your/hugginface/cache:/root/.cache/huggingface/diffusers`

Mind that the host path (first path up to ":") might very well be the same as the second if you're using the same diffusers library on the host and you didn't modify `HF_HOME`.

Some models may require a huggingface token to be downloaded, you can get yours at https://huggingface.co/settings/tokens after registering for free on their website. You can then add the token to your env with `-e TOKEN=<YOUR_TOKEN>`.

**P.S:** Feel free to open an issue for any problem you may face during installation.

# Samples

The internet is full of these, but I felt I couldn't let this repo go without sharing a few of "my own".. 

<p align="center" width="100%">
    <img width="48%" src="assets/0.png">
    <img width="48%" src="assets/1.png">
</p>

Fixed seed, slightly change text input (thanks to @mronchetti for the cool  prompt):
<p align="center" width="100%">
    <img width="32%" src="assets/redlove.png">
    <img width="32%" src="assets/greenlove.png">
    <img width="32%" src="assets/bluelove.png">
</p>

Fixed seed, same input, increase `guidance_scale` (more "adherent" to text) with a step of 5:
<p align="center" width="100%">
    <img width="32%" src="assets/village_5_2.png">
    <img width="32%" src="assets/village_10_2.png">
    <img width="32%" src="assets/village_15_2.png">
</p>
<p align="center" width="100%">
    <img width="48%" src="assets/village_0.png">
    <img width="48%" src="assets/village_5.png">
</p>

'Picture' vs 'Drawing' text input:
<p align="center" width="100%">
    <img width="48%" src="assets/3.png">
    <img width="48%" src="assets/4.png">
</p>


## TODO
 - [x] allow other input modalities (images)
 - [ ] support extra v2 features (depth-based generation, upscaling) 
 - [x] move model to specifiec GPU number (env variable)
 - [x] multi-gpu support (data parallel)
 - [x] multi-gpu support (PipelineParallel/model parallel)
 - [ ] Data+Model parallel: optimize memory assignment for 512x512 inference
 - [ ] dump and clear prompt history
 - [ ] test on older cudnn
