# LLM-driven Data Engineering

## Accounts to Follow

People
- [Li Yin](https://www.linkedin.com/in/li-yin-ai)
- [Chip Huyen](https://www.linkedin.com/in/chiphuyen/)
- [Zach Wilson](https://www.linkedin.com/in/eczachly)

Libraries
- [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow)
- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/run-llama/llama_index)

## Getting Started

Make an OpenAI account [here](https://platform.openai.com/) and then generate an API Key.
For Day 4, you'll need a [Pinecone](https://www.pinecone.io) account and API key. 

- Day 1 (LLM-driven data engineering
  - Lecture Video is [here](https://www.dataexpert.io/lesson/large-language-models-day-1-lecture)
  - Lab video is [here](https://www.dataexpert.io/lesson/large-language-models-day-1-lab)
- Day 2 (LLM dev with LangChain)
  - Lecture Video is [here](https://www.dataexpert.io/lesson/large-language-models-day-2-lecture)
  - Lab Video is [here](https://www.dataexpert.io/lesson/large-language-models-day-2-lab)
- Day 3 (Using LLM to provide business value)
  - Auto Feedback Repo [here](https://github.com/DataExpert-io/auto-feedback-example)
  - Lecture Video is [here](https://www.dataexpert.io/lesson/machine-learning-day-1-lecture-v4)
  - Lab Video is [here](https://www.dataexpert.io/lesson/machine-learning-day-1-lab-v4)
- Day 4 (Creating ZachGPT with RAG)
  - Vector Database Repo [here](https://github.com/DataExpert-io/vector-database-example)
  - Lecture Video is [here](https://www.dataexpert.io/lesson/machine-learning-day-2-lecture-v4)
  - Lab Video is [here](https://www.dataexpert.io/lesson/machine-learning-day-2-lab-v4)

## Setup

This project use [PostgreSQL](https://www.postgresql.org/download/macosx/)

Store the API key as an environment variable like:
`export OPENAI_API_KEY=<your_api_key>`
Or set it in Windows

The easiest way to install the dependencies is uv. [Install](https://docs.astral.sh/uv/getting-started/installation/) it.

Run the command `uv sync` to install the python environment and all of the libraries under `.venv` folder.

You should configure your IDE to select the interpreter under the .venv folder, or activate it through the command on your terminal:
```sh
source .venv/bin/activate
```

PS: If you don't want to use uv, run
```sh
pip install .
```

## Day 1 Lab

We'll be using the schemas from Dimensional Data Modeling Week 1 and generating the queries from the homework and labs except this time we'll do it via LLMs

## Day 2 Lab

We'll be using Langchain to auto generate SQL queries for us based on tables and writing LinkedIn posts in Zach Wilson's voice
### Setup

If you are watching live, you will be given a cloud database URL to use.
`export LANGCHAIN_DATABASE_URL=<value zach gives in Zoom>`

If you aren't watching live, you'll need to use the `halo_data_dump.dump` file located in the `data` folder

Running `pg_restore` with your local database should get you up and running pretty quickly. 

- example command, assuming you got Postgres up and running either via Homebrew or Docker:
 - `pg_restore -h localhost -p 5432  -d postgres -U <your laptop username> halo_data_dump.dump`

## Day 3 Lab

This lab leverages this [repo](https://github.com/DataExpert-io/auto-feedback-example)

## Day 4 Lab
This lab leverages this [repo](https://github.com/DataExpert-io/vector-database-example)

Add it to the environment `export PINECONE_API_KEY=<your pinecone API key>`


