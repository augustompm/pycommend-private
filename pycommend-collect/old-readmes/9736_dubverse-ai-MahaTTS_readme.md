<div align="center">

<a href="https://ibb.co/wN1LS7K"><img src="https://i.ibb.co/TB5T41H/Screenshot-2024-01-15-at-8-14-08-PM.png" alt="MahaTTS by Dubverse.ai" border="0" width=250></a>
<h1>MahaTTS: An Open-Source Large Speech Generation Model</h1>
a <a href = "https://black.dubverse.ai">Dubverse Black</a> initiative <br> <br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qkZz2km-PX75P0f6mUb2y5e-uzub27NW?usp=sharing)
</div>

------
## Update

Currently we are working on making our TTS system more robust. Latency is still an issue. Updates could take time.

## Next Up

We are currently training our large scale model. This will be a 1B parameter model, trained on 20K hours of data in 15 languages with 10 Indic Languages. 

## Description

MahaTTS, with Maha signifying 'Great' in Sanskrit, is a Text to Speech Model developed by [Dubverse.ai](https://dubverse.ai). We drew inspiration from the [Tortoise TTS](https://github.com/neonbjb/tortoise-tts) model, but our model uniquely utilizes seamless M4t wav2vec2 for semantic token extraction. As this specific variant of wav2vec2 is trained on multilingual data, it enhances our model's scalability across different languages.

We are providing access to pretrained model checkpoints, which are ready for inference and available for commercial use.

### Capabilities
Within a single model,
- generate voices in multiple seen and unseen speaker identities (voice cloning)
- generate voices in multiple langauges (multilingual and cross-lingual voice cloning)
- copy the style of speech from one speaker to another (cross-lingual voice cloning with prosody and intonation transfer)

### MahaTTS Architecture

<img width="993" alt="MahaTTS Architecture" src="https://github.com/dubverse-ai/MahaTTS/assets/32906806/7429d3b6-3f19-4bd8-9005-ff9e16a698f8">



## Updates

**7-01-2024**

- Smolie English (`smolie-en`) and Smolie Indic (`smolie-in`) released!

**13-11-2023**

- MahaTTS Open Sourced!


## Installation

```bash
pip install git+https://github.com/dubverse-ai/MahaTTS.git
```

```bash
pip install maha-tts
```

## api usage

```bash
#download example speakers ref files to copy the prosody from
!wget https://huggingface.co/Dubverse/MahaTTS/resolve/main/maha_tts/pretrained_models/infer_ref_wavs.zip
!unzip ./infer_ref_wavs.zip

import torch, glob
from maha_tts import load_models,infer_tts,config
from scipy.io.wavfile import write
from IPython.display import Audio,display

# PATH TO THE SPEAKERS WAV FILES
speaker =['/content/infer_ref_wavs/2272_152282_000019_000001/',
          '/content/infer_ref_wavs/2971_4275_000049_000000/',
          '/content/infer_ref_wavs/4807_26852_000062_000000/',
          '/content/infer_ref_wavs/6518_66470_000014_000002/']
```

### Inferring `smolie-en`, the English Model
```bash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diff_model,ts_model,vocoder,diffuser = load_models('Smolie-en',device)
print('Using:',device)

speaker_num = 0 # @param ["0", "1", "2", "3"] {type:"raw"}
text = "I freakin love how Elon came to life the moment they started talking about gaming and specifically diablo, you can tell that he didn't want that part of the discussion to end, while Lex to move on to the next subject! Once a true gamer, always a true gamer!" # @param {type:"string"}

ref_clips = glob.glob(speaker[speaker_num]+'*.wav')
audio,sr = infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder)

write('/content/test.wav',sr,audio)
```

### Inferring `smolie-in`, the Indic Multilingual Model
```bash
# SMOLIE-IN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diff_model,ts_model,vocoder,diffuser = load_models('Smolie-in',device)
print('Using:',device)

speaker_num = 0 # @param ["0", "1", "2", "3"] {type:"raw"}
text = "शाम के समय, आसमान में बिखरी हुई रंग-बिरंगी रौशनी से सजा हुआ नगर दृश्य एक रोमांटिक माहौल बना रहा था।" # @param {type:"string"}

langauge = 'hindi' # ['hindi','english','tamil', 'telugu', 'punjabi', 'marathi', 'gujarati', 'bengali', 'assamese']
language = torch.tensor(config.lang_index[langauge]).to(device).unsqueeze(0)

ref_clips = glob.glob(speaker[speaker_num]+'*.wav')
audio,sr = infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder,language)

write('/content/test.wav',sr,audio)
```

## Roadmap
- [x] Smolie English (`smolie-en`): Trained on 9k hours of English Podcast data
- [x] Smolie Indic (`smolie-in`): Trained on 400 hour of IIT Madras TTS audio data across 9 Indian languages
- [ ] Smolie Indic + English: Trained on big data (coming soon!)
- [ ] Optimizations for inference (looking for contributors, check issues)

## Sample Outputs
https://github.com/dubverse-ai/MahaTTS/assets/33093945/e6bde707-1e75-455f-b54d-fb1791c2afbf


## Technical Details

### Model Params
|      Model (Smolie)       | Parameters | Model Type |       Output      |  
|:-------------------------:|:----------:|------------|:-----------------:|
|   Text to Semantic (M1)   |    84 M    | Causal LM  |   10,001 Tokens   |
|  Semantic to MelSpec(M2)  |    430 M   | Diffusion  |   2x 80x Melspec  |
|      Hifi Gan Vocoder     |    13 M    |    GAN     |   Audio Waveform  |

### Languages Supported
| Language | Status |
| --- | :---: |
| English (en) | ✅ |
| Hindi (in) | ✅ |
| Indian English (in) | ✅ |
| Bengali (in) | ✅ |
| Tamil (in) | ✅ |
| Telugu (in) | ✅ |
| Punjabi (in) | ✅ |
| Marathi (in) | ✅ |
| Gujarati (in) | ✅ |
| Assamese (in) | ✅ |

## License

MahaTTS is licensed under the Apache 2.0 License. 

## 🙏 Appreciation

- [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
- [M4t Seamless](https://github.com/facebookresearch/seamless_communication) [AudioLM](https://arxiv.org/abs/2209.03143) and many other ground-breaking papers that enabled the development of MahaTTS
- [Diffusion training](https://github.com/openai/guided-diffusion) for training diffusion model
- [Huggingface](https://huggingface.co/docs/transformers/index) for related training and inference code
