# OuteTTS

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-OuteTTS_0.3_1B-blue)](https://huggingface.co/OuteAI/OuteTTS-0.3-1B)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-OuteTTS_0.3_1B_GGUF-blue)](https://huggingface.co/OuteAI/OuteTTS-0.3-500M-1B)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-OuteTTS_0.3_500M-blue)](https://huggingface.co/OuteAI/OuteTTS-0.3-500M)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-OuteTTS_0.3_500M_GGUF-blue)](https://huggingface.co/OuteAI/OuteTTS-0.3-500M-GGUF)
[![PyPI](https://img.shields.io/badge/PyPI-outetts-5c6c7a)](https://pypi.org/project/outetts/)
[![npm](https://img.shields.io/badge/npm-outetts-734440)](https://www.npmjs.com/package/outetts)

🤗 [Hugging Face](https://huggingface.co/OuteAI) | 💬 [Discord](https://discord.gg/vyBM87kAmf) | 𝕏 [X (Twitter)](https://twitter.com/OuteAI) | 🌐 [Website](https://www.outeai.com) | 📰 [Blog](https://www.outeai.com/blog)

OuteTTS is an experimental text-to-speech model that uses a pure language modeling approach to generate speech, without architectural changes to the foundation model itself.

## Compatibility

OuteTTS supports the following backends:

| **Backend**                 |
|-----------------------------|
| [Hugging Face Transformers](https://github.com/huggingface/transformers) |
| [GGUF llama.cpp](https://github.com/ggerganov/llama.cpp)              |
| [ExLlamaV2](https://github.com/turboderp/exllamav2)                   |
| [Transformers.js](https://github.com/huggingface/transformers.js)          |

## Installation

#### Python

```bash
pip install outetts
```

**Important:**
- For GGUF support, install `llama-cpp-python` manually. [Installation Guide](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation)
- For EXL2 support, install `exllamav2` manually. [Installation Guide](https://github.com/turboderp/exllamav2?tab=readme-ov-file#installation)

#### Node.js / Browser

```bash
npm i outetts
```

## Usage

### Interfaces

`outetts` package provide two interfaces for OuteTTS with support for different models:

| **Interface** | **Supported Models**                                     | **Documentation**                                              |
|---------------|---------------------------------------------------------|---------------------------------------------------------------|
| Interface v1  | OuteTTS-0.2, OuteTTS-0.1                                | [View Documentation](./docs/interface_v1_usage.md)            |
| Interface v2  | OuteTTS-0.3                                             | [View Documentation](./docs/interface_v2_usage.md)            |

**Generation Performance:** The model performs best with 30-second generation batches. This window is reduced based on the length of your speaker samples. For example, if the speaker reference sample is 10 seconds, the effective window becomes approximately 20 seconds. 

## Speaker Profile Recommendations

To achieve the best results when creating a speaker profile, consider the following recommendations:

1. **Audio Clip Duration:**
   - Use an audio clip of around **10 seconds**.
   - This duration provides sufficient data for the model to learn the speaker's characteristics while keeping the input manageable.

2. **Audio Quality:**
   - Ensure the audio is **clear and noise-free**. Background noise or distortions can reduce the model's ability to extract accurate voice features.

3. **Speaker Familiarity:**
   - The model performs best with voices that are similar to those seen during training. Using a voice that is **significantly different from typical training samples** (e.g., unique accents, rare vocal characteristics) might result in inaccurate replication.
   - In such cases, you may need to **fine-tune the model** specifically on your target speaker's voice to achieve a better representation.

4. **Parameter Adjustments:**
   - Adjust parameters like `temperature` in the `generate` function to refine the expressive quality and consistency of the synthesized voice.

## Credits

- WavTokenizer: [GitHub Repository](https://github.com/jishengpeng/WavTokenizer)
  - `decoder` and `encoder` folder files are from this repository
- CTC Forced Alignment: [PyTorch Tutorial](https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)
- Uroman: [GitHub Repository](https://github.com/isi-nlp/uroman)
    - "This project uses the universal romanizer software 'uroman' written by Ulf Hermjakob, USC Information Sciences Institute (2015-2020)".
- mecab-python3 [GitHub Repository](https://github.com/SamuraiT/mecab-python3)
