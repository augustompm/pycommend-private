
# OLMo-Eval

OLMo-Eval is a repository for evaluating open language models.


## Note of Deprecation

**NOTE:** This repository has been superceded by the OLMES repository, available 
at https://github.com/allenai/olmes (Open Language Model Evaluation System).


## Overview

The `olmo_eval` framework is a way to run evaluation pipelines for language models on NLP tasks. 
The codebase is extensible and contains `task_sets` and example configurations, which run a series
of [`tango`](https://github.com/allenai/tango) steps for computing the model outputs and metrics.


Using this pipeline, you can evaluate _m_ models on _t_ task_sets, where each task_set consists of one or more individual tasks.
Using task_sets allows you to compute aggregate metrics for multiple tasks. The optional `google-sheet` integration can be used
for reporting.

The pipeline is built using [ai2-tango](https://github.com/allenai/tango) and [ai2-catwalk](https://github.com/allenai/catwalk).

## Installation


After cloning the repository, please run

```commandline
conda create -n eval-pipeline python=3.10
conda activate eval-pipeline
cd OLMo-Eval
pip install -e .
```


## Quickstart

The current `task_sets` can be found at [configs/task_sets](configs/task_sets). In this example, we run `gen_tasks` on `EleutherAI/pythia-1b`. The example config is [here](configs/example_config.jsonnet).

The configuration can be run as follows:

```commandline
tango --settings tango.yml run configs/example_config.jsonnet --workspace my-eval-workspace
```

This executes all the steps defined in the config, and saves them in a local `tango` workspace called `my-eval-workspace`. If you add a new task_set or model to your config and run the same command again, it will reuse the previous outputs, and only compute the new outputs.

The output should look like this:

<img width="1886" alt="Screen Shot 2023-12-04 at 9 22 35 PM" src="https://github.com/allenai/ai2-llm-eval/assets/6500683/14a74e61-75d8-470c-8bde-12e35c38c44a">

New models and datasets can be added by modifying the [example configuration](configs/example_config.jsonnet).

### Load pipeline output

```python
from tango import Workspace
workspace = Workspace.from_url("local://my-eval-workspace")
result = workspace.step_result("combine-all-outputs")
```

Load individual task results with per instance outputs

```python
result = workspace.step_result("outputs_pythia-1bstep140000_gen_tasks_drop")
```


## Evaluating common models on standard benchmarks

The [eval_table](configs/eval_table.jsonnet) config evaluates `falcon-7b`, `mpt-7b`, `llama2-7b`, and `llama2-13b`, on [`standard_benchmarks`](configs/task_sets/standard_benchmarks.libsonnet) and [`MMLU`](configs/task_sets/mmlu_tasks.libsonnet). Run as follows:


```commandline
tango --settings tango.yml run configs/eval_table.jsonnet --workspace my-eval-workspace
```


## PALOMA

This repository was also used to run evaluations for the [PALOMA paper](https://www.semanticscholar.org/paper/Paloma%3A-A-Benchmark-for-Evaluating-Language-Model-Magnusson-Bhagia/1a3f7e23ef8f0bf06d0efa0dc174e4e361226ead?utm_source=direct_link)

Details on running the evaluation on PALOMA can be found [here](paloma/README.md).


## Advanced

* [Save output to google sheet](ADVANCED.md#save-output-to-google-sheet)
* [Use a remote workspace](ADVANCED.md#use-a-remote-workspace)
* [Run without Tango (useful for debugging)](ADVANCED.md#run-without-tango)
* [Run on Beaker](BEAKER.md)


