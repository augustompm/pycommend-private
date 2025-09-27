# Charlie Mnemonic

As part of our research efforts in continual learning, we are open-sourcing **Charlie Mnemonic, the first personal** assistant (LLM agent) equipped with **Long-Term Memory (LTM)**. 

At first glance, Charlie might resemble existing LLM agents like ChatGPT, Claude, and Gemini. However, its distinctive feature is the implementation of LTM, enabling it to **learn from every interaction**. This includes **storing and integrating user messages, assistant responses, and environmental feedback into LTM** for future retrieval when relevant to the task at hand.

Charlie Mnemonic employs a combination of **Long-Term Memory (LTM), Short-Term Memory (STM)**, and **episodic memory** to deliver **context-aware responses**. This ability to remember interactions over time significantly improves the coherence and personalization of conversations.

Read more [on our blog](https://www.goodai.com/introducing-charlie-mnemonic/)

## Features

- User authentication and session management.
- Continual learning through user interactions.
- Ability to generate and process audio messages.
- User data management including import/export of data. (WIP)
- Extensible with additional modules (addons).
- WebSocket support for real-time communication.
- Use local Ollama instance instead of OpenAI.

## Stoic Mentor
We’ve included a new personality, a stoic mentor.
You can enable the Stoic Mentor personality by going to the Settings Menu and changing the system prompt to the ‘Stoic’ preset. The agent will be prompted to help you navigate through life, find clarity and resilience through thoughtful conversation. Your feedback to this new personality, and how it may be improved, is highly appreciated!

## Installation

- How to install and run Charlie Mnemonic on [Windows](docs/WININSTALL.md), [Mac](docs/MACINSTALL.md) or [Linux](docs/LINUXINSTALL.md)


- How to install, develop and run Charlie Mnemonic [manually](docs/DEV-SETUP.md)

- To use Ollama please look at [Ollama](https://ollama.com/) to download and setup the Ollama server.

## Links

- [Contributing](docs/CONTRIBUTING.md)
- [License](docs/LICENSE.md)
- [API Endpoints](docs/API.md)
- [Persistence](docs/PERSISTENCE.md)
- [Security](docs/SECURITY.md)
- [Licenses of used libraries](docs/LICENSES.txt)
- Charlie Mnemonic on [docker hub](https://hub.docker.com/r/goodaidev/charlie-mnemonic)