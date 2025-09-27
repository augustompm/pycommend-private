---
title: Omni-Zero
emoji: 🧛🏻‍♂️
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: gpl-3.0
---

## **Announcement:** Omni-Zero Couples is now live
[https://github.com/okaris/omni-zero-couples](https://github.com/okaris/omni-zero-couples)

[![Buy me a coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=vk654cf2pv8&button_colour=BD5FFF&font_colour=ffffff&font_family=Bree&outline_colour=000000&coffee_colour=FFDD00)](https://www.buymeacoffee.com/vk654cf2pv8)


# Omni-Zero: A diffusion pipeline for zero-shot stylized portrait creation. 
- [x] Release single person code
- [x] Build Gradio app
- [x] Add Huggingface Spaces ZeroGPU demo
- [x] Add Replicate demo
- [x] Add Replicate cog example
- [x] Add fal.ai Workflow v2 (LoRA support)
- [x] Release couples code
- [ ] Add LoRA support

## Try our free demo on StyleOf [https://styleof.com/s/remix-yourself](https://styleof.com/s/remix-yourself)
![Omni-Zero](https://github.com/okaris/omni-zero/assets/1448702/2423a219-2191-4b6a-8e7b-43230e137cd7)

## Use Omni-Zero in [fal.ai](https://fal.ai) Workflows [https://fal.ai/dashboard/workflows/okaris/omni-zero](https://fal.ai/dashboard/workflows/okaris/omni-zero)
![Omni-Zero](https://github.com/okaris/omni-zero/assets/1448702/2ccbdf24-eb41-4a85-975e-af701fc4a879)

## V2 of [fal.ai](https://fal.ai) Workflow that supports custom LoRAs and more [http://fal.ai/dashboard/workflows/okaris/omni-zero-v2](http://fal.ai/dashboard/workflows/okaris/omni-zero-v2) ![Omni-Zero](https://github.com/okaris/omni-zero/assets/1448702/b7c3178b-d592-4287-a303-619f31599a06)

## Run on Replicate [https://replicate.com/okaris/omni-zero](https://replicate.com/okaris/omni-zero)
![Omni-Zero](https://github.com/okaris/omni-zero/assets/1448702/0d53489b-89eb-4277-907f-4317cc98db74)

## Use Omni-Zero in HuggingFace Spaces ZeroGPU [https://huggingface.co/spaces/okaris/omni-zero](https://huggingface.co/spaces/okaris/omni-zero)
![Omni-Zero](https://github.com/okaris/omni-zero/assets/1448702/1d4c40e0-41c5-4127-ba06-aec52a2d179d)

### Single Identity and Style
![Omni-Zero](https://github.com/okaris/omni-zero/assets/1448702/2c51fb77-a810-4c0a-9555-791a294455ca)

### Multiple Identities and Styles (WIP)
![Frame 7-3](https://github.com/okaris/omni-zero/assets/1448702/c5c20961-83bc-47f7-86ed-5948d5590f07)

### How to run
```
git clone --recursive https://github.com/okaris/omni-zero.git
cd omni-zero
pip install -r requirements.txt
python demo.py
```

### Credits
- Special thanks to [fal.ai](https://fal.ai) for providing compute for the research and hosting
- This project wouldn't be possible without the great work of the [InstantX Team](https://github.com/InstantID)
- Thanks to [@fofrAI](http://twitter.com/fofrAI) for inspiring me with his [face-to-many workflow](https://github.com/fofr/cog-face-to-many)
- Thanks to Matteo ([@cubiq](https://twitter.com/cubiq])) for creating the ComfyUI nodes for IP-Adapter
