# SynCamMaster: Synchronizing Multi-Camera Video Generation from Diverse Viewpoints

<div align="center">
<div align="center" style="margin-top: 0px; margin-bottom: 0px;">
<img src=https://github.com/user-attachments/assets/b33c5b67-3881-4fa3-b853-f932eebc9c50 width="50%"/>
</div>

### [<a href="https://arxiv.org/abs/2412.07760" target="_blank">arXiv</a>] [<a href="https://jianhongbai.github.io/SynCamMaster/" target="_blank">Project Page</a>] [<a href="https://huggingface.co/datasets/KwaiVGI/SynCamVideo-Dataset/" target="_blank">Dataset</a>]

_**[Jianhong Bai<sup>1*</sup>](https://jianhongbai.github.io/), [Menghan Xia<sup>2†</sup>](https://menghanxia.github.io/), [Xintao Wang<sup>2</sup>](https://xinntao.github.io/), [Ziyang Yuan<sup>3</sup>](https://scholar.google.ru/citations?user=fWxWEzsAAAAJ&hl=en), [Xiao Fu<sup>4</sup>](https://fuxiao0719.github.io/), <br>[Zuozhu Liu<sup>1</sup>](https://person.zju.edu.cn/en/lzz), [Haoji Hu<sup>1</sup>](https://person.zju.edu.cn/en/huhaoji), [Pengfei Wan<sup>2</sup>](https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl=en), [Di Zhang<sup>2</sup>](https://openreview.net/profile?id=~Di_ZHANG3)**_
<br>
(*Work done during an internship at KwaiVGI, Kuaishou Technology †corresponding author)

<sup>1</sup>Zhejiang University, <sup>2</sup>Kuaishou Technology, <sup>3</sup>Tsinghua University, <sup>4</sup>CUHK.

</div>

## 📖 Introduction

**TL;DR:** We propose SynCamMaster, an efficient method to lift pre-trained text-to-video models for open-domain multi-camera video generation from diverse viewpoints.  <br>

https://github.com/user-attachments/assets/1ecfaea8-5d87-4bb5-94fc-062f84bd67a1

## 🔥 Updates
- __[2024.12.10]__: Release the [project page](https://jianhongbai.github.io/SynCamMaster/) and the [SynCamVideo Dataset](https://huggingface.co/datasets/KwaiVGI/SynCamVideo-Dataset/).

## 📷 SynCamVideo Dataset
### 1. Dataset Introduction
The SynCamVideo Dataset is a multi-camera synchronized video dataset rendered using the Unreal Engine 5. It consists of 1,000 different scenes, each captured by 36 cameras, resulting in a total of 36,000 videos. SynCamVideo features 50 different animals as the "main subjects" and utilizes 20 different locations from [Poly Haven](https://polyhaven.com/hdris) as backgrounds. In each scene, 1-2 subjects are selected from the 50 animals and move along a predefined trajectory, the background is randomly chosen from the 20 locations, the 36 cameras simultaneously record the subjects' movements.

The cameras in each scene are placed on a hemispherical surface at a distance to the scene center of 3.5m - 9m. To ensure the rendered videos have minimal domain shift with real-world videos, we constraint the elevation of each camera between 0° - 45°, and the azimuth between 0° - 360°. Each camera is randomly sampled within the constraints described above, rather than using the same set of camera positions across scenes. The figure below shows an example, where the red star indicates the center point of the scene (slightly above the ground), and the videos are rendered from the synchronized cameras to capture the movements of the main subjects (a goat and a bear in the case).

![3_resized](https://github.com/user-attachments/assets/01e2671c-9fa2-4290-b43e-14905b4c9685)

The SynCamVideo Dataset can be used to train multi-camera synchronized video generation models, inspiring applications in areas such as filmmaking and multi-view data generation for downstream tasks.

### 2. File Structure
```
SynCamVideo
├── train
│   ├── videos    # training videos
│   │   ├── scene1    # one scene
│   │   │   ├── xxx.mp4    # synchronized 100-frame videos at 480x720 resolution
│   │   │   └── ...
│   │   │   ...
│   │   └── scene1000
│   │       ├── xxx.mp4
│   │       └── ...
│   └── cameras    # training cameras
│       ├── scene1    # one scene
│       │   └── xxx.json    # extrinsic parameters corresponding to the videos
│       │   ...
│       └── scene1000
│           └── xxx.json
└──val
    └── cameras    # validation cameras
        ├── Hemi36_4m_0    # distance=4m, elevation=0°
        │   └── Hemi36_4m_0.json    # 36 cameras: distance=4m, elevation=0°, azimuth=i * 10°
        │   ...
        └── Hemi36_7m_45
            └── Hemi36_7m_45.json
```

### 3. Useful scripts
- Camera Visualization
```bash
python vis_cam.py --pose_file_path ./val/cameras/Hemi36_4m_0/Hemi36_4m_0_transforms.json --num_cameras 36
```

The visualization script is modified from [CameraCtrl](https://github.com/hehao13/CameraCtrl/blob/main/tools/visualize_trajectory.py), thanks for their inspiring work.

![4](https://github.com/user-attachments/assets/2c588f2a-e143-4c64-8fe4-a1f1268cf25b)


## 🏁 Getting Started (SynCamMaster+CogVideoX)

**Note:** The model we used in our paper is an internal research propose T2V model, not CogVideoX. Due to company policy restrictions, we are unable to open-source the model used in the paper. Therefore, we migrated SynCamMaster to CogVideoX to validate the effectiveness of our method. As a result, due to the differences in the base T2V model, you may not be able to achieve the same results as demonstrated in the demo.

### 1. Environment Set Up
Our environment setup is identical to [CogVideoX](https://github.com/THUDM/CogVideo). You can refer to their configuration to complete your environment setup.
```bash
conda create -n syncammaster python=3.10
conda activate syncammaster
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

**TODO: upload the pre-trained checkpoints.**

### 3. Code Snapshot
The following [code](https://github.com/KwaiVGI/SynCamMaster/blob/main/syncammaster/transformer_3d.py#L120-L138) showcases the core components of SynCamMaster, namely the camera encoder, multi-view attention layer, and a linear projector within each transformer block, as demonstrated in Fig. 2 of our [paper](https://arxiv.org/abs/2412.07760).
```python
# 1. add pose feature
pose = rearrange(pose, "b v d -> (b v) 1 d")
pose_embedding = self.cam_encoder(pose)
norm_hidden_states = norm_hidden_states + pose_embedding

# 2. multi-view attention
norm_hidden_states = rearrange(norm_hidden_states, "(b v) (f s) d -> (b f) (v s) d", f=frame_num, v=view_num)
norm_encoder_hidden_states = rearrange(norm_encoder_hidden_states, "(b v) n d -> b (v n) d", v=view_num)
norm_encoder_hidden_states = repeat(norm_encoder_hidden_states, "b n d -> (b f) n d", f=frame_num)
attn_hidden_states, _ = self.attn_syncam(
    hidden_states=norm_hidden_states,
    encoder_hidden_states=norm_encoder_hidden_states,
    image_rotary_emb=image_rotary_emb_view,
)

# 3. project back with residual connection
attn_hidden_states = self.projector(attn_hidden_states)
attn_hidden_states = rearrange(attn_hidden_states, "(b f) (v s) d -> (b v) (f s) d", f=frame_num, v=view_num)
hidden_states = hidden_states + gate_msa * attn_hidden_states
```

## 🚀 Inference (SynCamMaster+CogVideoX)

```bash
python syncammaster_inference.py --model_path THUDM/CogVideoX-2b
```


## 🤗 Awesome Related Works
Feel free to explore these outstanding related works, including but not limited to:

[GCD](https://gcd.cs.columbia.edu/): synthesize large-angle novel viewpoints of 4D dynamic scenes from a monocular video.

[CVD](https://collaborativevideodiffusion.github.io): multi-view video generation with multiple camera trajectories.

[SV4D](https://sv4d.github.io): multi-view consistent dynamic 3D content generation.

Additionally, check out our "MasterFamily" projects:

[3DTrajMaster](http://fuxiao0719.github.io/projects/3dtrajmaster): control multiple entity motions in 3D space (6DoF) for text-to-video generation.

[StyleMaster](https://zixuan-ye.github.io/stylemaster/): enable artistic video generation and translation with reference style image.


## Acknowledgments
We thank Jinwen Cao, Yisong Guo, Haowen Ji, Jichao Wang, and Yi Wang from Kuaishou Technology for their invaluable help in constructing the SynCamVideo-Dataset. We thank [Guanjun Wu](https://guanjunwu.github.io/) and Jiangnan Ye for their help on running 4DGS.

## 🌟 Citation

Please leave us a star 🌟 and cite our paper if you find our work helpful.
```
@misc{bai2024syncammaster,
      title={SynCamMaster: Synchronizing Multi-Camera Video Generation from Diverse Viewpoints}, 
      author={Jianhong Bai and Menghan Xia and Xintao Wang and Ziyang Yuan and Xiao Fu and Zuozhu Liu and Haoji Hu and Pengfei Wan and Di Zhang},
      year={2024},
      eprint={2412.07760},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.07760}, 
}
```
