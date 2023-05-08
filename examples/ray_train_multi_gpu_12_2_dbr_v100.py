# Databricks notebook source
# MAGIC %md
# MAGIC **This notebook has been tested on 12.X ML DBR runtime using 2 p3dn.48xlarge**

# COMMAND ----------

# MAGIC %md
# MAGIC #Fine-Tuning with Ray AIR and DeepSpeed on Databricks
# MAGIC
# MAGIC In this example, we will showcase how to use the Ray AIR for **pythia-12b**. These causal language model trained on the Pile dataset(825 GB). This particular model has 12 billion parameters. For more information on pythia-12b click [here](https://huggingface.co/EleutherAI/pythia-12b).
# MAGIC
# MAGIC We will use Ray AIR (with the 🤗 Transformers integration) and a pretrained model from Hugging Face hub. Note that you can easily adapt this example to use other similar models.
# MAGIC
# MAGIC This example focuses more on the performance and distributed computing aspects of Ray AIR. 
# MAGIC
# MAGIC It is highly recommended to read [Ray AIR Key Concepts](https://raw.githubusercontent.com/ray-project/ray/master/doc/source/ray-air/examples/air-key-concepts) and [Ray Data Key Concepts](https://raw.githubusercontent.com/ray-project/ray/master/doc/source/ray-air/examples/data_key_concepts) before starting this example.
# MAGIC
# MAGIC ```{note}
# MAGIC In order to run this example, make sure your Ray cluster has access to at least 8 GPU's with 24 or more GBs of memory. The amount of memory needed will depend on the model. This notebook has been  tested with 2 p3dn.48xlarge workers and g4dn.8xlarge head node.
# MAGIC ```
# MAGIC Benefits:
# MAGIC
# MAGIC - No command-line trigger / and provides real-time updates
# MAGIC - Better UI to understand JOB performance and adjust the batch-size and performance configd
# MAGIC - Ray data API support data in parquet,csv and hf format
# MAGIC - Has MLFlow integration to track Experiments
# MAGIC
# MAGIC In this notebook, we will:
# MAGIC 1. [Add Dependencies to run deepspeed](#Deepspeed)
# MAGIC 2. [Set up Ray](#setup)
# MAGIC 3. [Load the dataset](#load)
# MAGIC 4. [Preprocess the dataset with Ray AIR](#preprocess)
# MAGIC 5. [Run the training with Ray AIR](#train)
# MAGIC 6. [Generate text from prompt with Ray AIR](#predict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dependencies for Deepspeed <a name="deepspeed"></a>
# MAGIC Uncomment and run the following line in order to create an init script which loads the dependencies required for Deepspeed (this notebook is being tested with `transformers==4.26.0`):

# COMMAND ----------

# kernel_gateway_init = """
# #!/bin/bash

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb -O /tmp/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb && \
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-3_11.5.1.109-1_amd64.deb -O /tmp/libcublas-dev-11-3_11.5.1.109-1_amd64.deb && \
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb -O /tmp/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb && \
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-3_10.2.4.109-1_amd64.deb -O /tmp/libcurand-dev-11-3_10.2.4.109-1_amd64.deb && \
# dpkg -i /tmp/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb && \
# dpkg -i /tmp/libcublas-dev-11-3_11.5.1.109-1_amd64.deb && \
# dpkg -i /tmp/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb && \
# dpkg -i /tmp/libcurand-dev-11-3_10.2.4.109-1_amd64.deb
# """ 
# # Change ‘username’ to your Databricks username in DBFS
# # Example: username = “stephen.offer@databricks.com”
# username = "puneet.jain@databricks.com"
# dbutils.fs.put("dbfs:/Users/{0}/init/ray.sh".format(username), kernel_gateway_init, True)
# "dbfs:/Users/{0}/init/ray.sh".format(username)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Ray <a name="setup"></a>
# MAGIC
# MAGIC First, Let us start a ray cluster based on the cluster configuration. we need to specify the number of cores and gpus available per worker to **setup_ray_cluster** to create the correct multi-node setup

# COMMAND ----------

#install dependencies
%pip install -r requirements.txt

# COMMAND ----------

# Import all the packages
import os
import re
import json
import logging
import subprocess
import mlflow

from pathlib import Path
from functools import partial
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import ray
from ray.air import session
import ray.util.scheduling_strategies
from ray.train.huggingface import HuggingFaceTrainer
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.config import ScalingConfig

from ray.data.preprocessors import Chain
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster,MAX_NUM_WORKER_NODES

from training.trainer import load_training_dataset
from training.consts import DEFAULT_INPUT_MODEL, SUGGESTED_INPUT_MODELS
from training.utils import get_or_create_experiment




from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from datasets import load_dataset,load_from_disk

import numpy as np
import pandas as pd
import torch 

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from training.consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY_NL,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define variables (can be added as databricks widgets as well)

# COMMAND ----------

pretrained_model_name_or_path = "EleutherAI/pythia-12b"
use_gpu = True
num_workers = 16
num_cpu_cores_per_worker = 96
num_gpu_per_worker = 8
max_length = 1024
local_output_dir = '/tmp/run/details'
gradient_checkpointing = True
seed = DEFAULT_SEED 
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
experiment_location = f"/Users/{username}/dolly_multi-gpu"

# COMMAND ----------

# shutdown_ray_cluster()

# COMMAND ----------

# Start the ray cluster
setup_ray_cluster(
  num_worker_nodes=MAX_NUM_WORKER_NODES,
  num_cpus_per_node=num_cpu_cores_per_worker,
  num_gpus_per_node=num_gpu_per_worker,
  collect_log_to_path="/dbfs/path/to/ray_collected_logs")

# COMMAND ----------

# MAGIC %md
# MAGIC We will use `ray.init()` to initialize the ray cluster in the current session.
# MAGIC
# MAGIC We define a `runtime_env` to ensure that the Ray workers have access to all the necessary packages. You can omit the `runtime_env` argument if you have all of the packages already installed on each node in your cluster.

# COMMAND ----------

runtime_env = {
    "env_vars": {"RAY_memory_monitor_refresh_ms": "0"}
}
ray.init(runtime_env=runtime_env)

# COMMAND ----------

# MAGIC %md
# MAGIC we will catch the models in the local nodes to avoid getting it from HF Server everytime during training calling the `snapshot_download()` command from HF 

# COMMAND ----------

# THIS SHOULD BE HIDDEN IN DOCS AND ONLY RAN IN CI
# Download the model from our S3 mirror as it's faster

def force_on_node(node_id: str, remote_func_or_actor_class):
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id, soft=False
    )
    options = {"scheduling_strategy": scheduling_strategy}
    return remote_func_or_actor_class.options(**options)


def run_on_every_node(remote_func_or_actor_class, **remote_kwargs):
    refs = []
    for node in ray.nodes():
        if node["Alive"] and node["Resources"].get("GPU", None):
            refs.append(
                force_on_node(node["NodeID"], remote_func_or_actor_class).remote(
                    **remote_kwargs
                )
            )
    return ray.get(refs)


@ray.remote(num_gpus=1)
def download_model():
    snapshot_download(pretrained_model_name_or_path,resume_download=True) 
  

_ = run_on_every_node(download_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the dataset <a name="load"></a>
# MAGIC
# MAGIC We will be fine-tuning the model on the the Databricks crowd sourced dataset , it comprised of 15,000 lines of Question and Answer pairs . The aim will be to make the GPT model.
# MAGIC
# MAGIC We will use [Ray Data](https://raw.githubusercontent.com/ray-project/ray/master/doc/source/ray-air/examples/data) for distributed preprocessing and data ingestion. We can easily convert the dataset obtained from Hugging Face Hub to Ray Data by using {meth}`ray.data.from_huggingface`.

# COMMAND ----------

# Splitting the data into test and train 

current_dataset = load_training_dataset()
current_dataset = current_dataset.train_test_split(seed=DEFAULT_SEED)

# current_dataset['train'].select(list(range(0,1000))).save_to_disk("train.hf")
# current_dataset['test'].select(list(range(0,1000))).save_to_disk('test.hf')

current_dataset['train'].save_to_disk("/local_disk0/train.hf")
current_dataset['test'].save_to_disk('/local_disk0/test.hf')
del current_dataset

# load the final data as ray data-set
train_dataset = ray.data.from_huggingface(load_from_disk('/local_disk0/train.hf'))
test_dataset = ray.data.from_huggingface(load_from_disk('/local_disk0/test.hf'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess the Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC We will need to do some preprocessing. For that, we will define two [Ray AIR Preprocessors](https://raw.githubusercontent.com/ray-project/ray/master/doc/source/ray-air/examples/air-preprocessors) using the {class}`~ray.data.preprocessors.BatchMapper` API, allowing us to define functions that will be applied on batches of data.
# MAGIC
# MAGIC The `preprocess` function will call The `tokenize` function will take the lines and tokenize them using the 🤗 Tokenizer associated with the model, ensuring each entry has the same length (`max_length`) by padding and truncating. This is necessary for training.
# MAGIC
# MAGIC ```{note}
# MAGIC This preprocessing can be done in other ways. A common pattern is to tokenize first, and then split the obtained tokens into equally-sized blocks.
# MAGIC ```

# COMMAND ----------

import ray
from pathlib import Path
from ray import tune
from datasets import Dataset
from ray.data.preprocessors import Chain, BatchMapper
from ray.air.util.check_ingest import DummyTrainer
from ray.air.config import ScalingConfig,RunConfig,CheckpointConfig


from training.trainer import load_tokenizer,preprocess_batch,\
                             DataCollatorForCompletionOnlyLM,get_model_tokenizer

def preprocess(batch):
  tokenizer = load_tokenizer(pretrained_model_name_or_path)
  seed=DEFAULT_SEED
  dataset = Dataset.from_pandas(batch)
  _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
  dataset = dataset.map(
      _preprocessing_function,
      batched=True,
      remove_columns=["instruction", "context", "response", "text", "category"],
  )

  # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
  dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
  dataset = dataset.shuffle(seed=seed)
  return dataset.to_pandas()


preprocessor = Chain(
    BatchMapper(preprocess, batch_format="pandas")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fine-tuning the model with Ray AIR <a name="train"></a>
# MAGIC
# MAGIC We can now configure Ray AIR's {class}`~ray.train.huggingface.huggingface_trainer.HuggingFaceTrainer` to perform distributed fine-tuning of the model. In order to do that, we specify a `trainer_init_per_worker` function, which creates a 🤗 Transformers `Trainer` that will be distributed by Ray using Distributed Data Parallelism (using PyTorch Distributed backend internally). This means that each worker will have its own copy of the model, but operate on different data, At the end of each step, all the workers will sync gradients.
# MAGIC
# MAGIC Because pythia-12b is a relatively large model, it may not be possible to fit it on smaller GPU types (<=16 GB GRAM). To deal with that issue, we can use [DeepSpeed](https://github.com/microsoft/DeepSpeed), a library to optimize the training process and allow us to (among other things) offload and partition optimizer and parameter states, reducing GRAM usage. Furthermore, DeepSpeed ZeRO Stage 3 allows us to load large models without running out of memory.
# MAGIC
# MAGIC 🤗 Transformers and Ray AIR's integration ({class}`~ray.train.huggingface.huggingface_trainer.HuggingFaceTrainer`) allow you to easily configure and use DDP and DeepSpeed. All you need to do is specify the DeepSpeed configuration in the [`TrainingArguments`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) object.
# MAGIC
# MAGIC ```{tip}
# MAGIC There are many DeepSpeed settings that allow you to trade-off speed for memory usage. The settings used below are tailored to the cluster setup used (16 g4dn.4xlarge nodes) and per device batch size of 16. Some things to keep in mind:
# MAGIC - If your GPUs support bfloat16, use that instead of float16 mixed precision to get better performance and prevent overflows. Replace `fp16=True` with `bf16=True` in `TrainingArguments`.
# MAGIC - If you are running out of GRAM: try reducing batch size (defined in the cell below the next one), set `"overlap_comm": False` in DeepSpeed config.
# MAGIC - If you are running out of RAM, add more nodes to your cluster, use nodes with more RAM, set `"pin_memory": False` in the DeepSpeed config, reduce the batch size, and remove `"offload_param"` from the DeepSpeed config.
# MAGIC
# MAGIC For more information on DeepSpeed configuration, refer to [Hugging Face documentation](https://huggingface.co/docs/transformers/main_classes/deepspeed) and [DeepSpeed documentation](https://www.deepspeed.ai/docs/config-json/).
# MAGIC
# MAGIC Additionally, if you prefer a lower-level API, the logic below can be expressed as an [Accelerate training loop](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/deepspeed_with_config_support.py) distributed by a Ray AIR {class}`~ray.train.torch.torch_trainer.TorchTrainer`.
# MAGIC ```
# MAGIC
# MAGIC #### Training speed
# MAGIC
# MAGIC As we are using data parallelism, each worker operates on its own shard of the data. The batch size set in `TrainingArguments` is the **per device batch size** (per worker batch size). By changing the number of workers, we can change the **effective batch size** and thus the time needed for training to complete. The effective batch size is then calculated as `per device batch size * number of workers * number of gradient accumulation steps`. As we add more workers, the effective batch size rises and thus we need less time to complete a full epoch. While the speedup is not exactly linear due to extra communication overheads, in many cases it can be close to linear.
# MAGIC
# MAGIC The preprocessed dataset has ~15000 examples. We have set per device batch size to 12.
# MAGIC
# MAGIC * With 2 p3dn.48xlarge nodes, the effective batch size was 192, which equals to ~79 steps per epoch.

# COMMAND ----------

def trainer_init_per_worker(train_dataset, eval_dataset=None, **config):

    set_seed(seed)

    # Use the actual number of CPUs assigned by Ray
    os.environ["OMP_NUM_THREADS"] = str(
        session.get_trial_resources().bundles[-1].get("CPU", 1)
    )
    # Enable tf32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True

    # Get config details

    epochs = config.get("epochs")
    lr = config.get("lr")
    per_device_train_batch_size = config.get("per_device_train_batch_size")
    per_device_eval_batch_size = config.get("per_device_eval_batch_size")
    logging_steps = config.get("logging_steps")
    save_strategy= config.get("save_strategy")
    evaluation_strategy = config.get("evaluation_strategy")
    save_steps = config.get("save_steps")
    eval_steps = config.get("eval_steps") 
    warmup_steps = config.get("warmup_steps")
    disable_tqdm=config.get("disable_tqdm")
    remove_unused_columns=config.get("remove_unused_columns")
    deepspeed=config.get("deepspeed", "configs/ds_z3_bf16_config.json")

    with open('/tmp'+'/deepspeed.json', 'w') as f:
      json.dump(deepspeed, f)

    print("Preparing training arguments")
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=True, # change to true if using v100
        bf16=False,# chenge to false if using v100
        learning_rate=lr,
        num_train_epochs=epochs,
        deepspeed='/tmp'+'/deepspeed.json',
        gradient_checkpointing=gradient_checkpointing,
        logging_strategy=evaluation_strategy,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        load_best_model_at_end=False,
        disable_tqdm=disable_tqdm,
        remove_unused_columns=remove_unused_columns,
        warmup_steps=warmup_steps)

    print("Loading model")

    model, tokenizer = get_model_tokenizer(
        pretrained_model_name_or_path=pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing
    )

    print("Model loaded")
    print("Train data size: %d", len(train_dataset))
    print("Test data size: %d", len(eval_dataset))

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    return trainer

# COMMAND ----------

# MAGIC %md
# MAGIC With our `trainer_init_per_worker` complete, we can now instantiate the {class}`~ray.train.huggingface.huggingface_trainer.HuggingFaceTrainer`. Aside from the function, we set the `scaling_config`, controlling the amount of workers and resources used, and the `datasets` we will use for training and evaluation.
# MAGIC
# MAGIC We pass the preprocessors we have defined earlier as an argument, wrapped in a {class}`~ray.data.preprocessors.chain.Chain`. The preprocessor will be included with the returned {class}`~ray.air.checkpoint.Checkpoint`, meaning it will also be applied during inference.

# COMMAND ----------

# get or create experiment
get_or_create_experiment(experiment_location)


root_path = os.getcwd()
deepspeed_config = os.path.join(root_path, "config/ds_z3_bf16_config.json")

with open(deepspeed_config) as json_data:
    deepspeed_config = json.load(json_data)

trainer = HuggingFaceTrainer(
    trainer_init_per_worker=trainer_init_per_worker,
    trainer_init_config={
        "deepspeed": deepspeed_config, 
        "lr" : 1e-6, # per device
        "per_device_train_batch_size" : 10,
        "per_device_eval_batch_size" : 10,
        "save_strategy" : "no",
        "evaluation_strategy" : "steps",
        "logging_steps" : 50,
        "save_steps" : 200,
        "eval_steps" : 50,
        "warmup_steps" : 25,
        "disable_tqdm" : True,
        "remove_unused_columns" :False,
        "epochs": 2},
        scaling_config=ScalingConfig(
        num_workers=16,
        use_gpu=use_gpu,
        resources_per_worker={"GPU": 1, "CPU": 10}), # should be total cores in node /total gpu's in node -2
    run_config = RunConfig(
                local_dir =  f"/dbfs/{username}/dolly_train/job/",
                callbacks=[MLflowLoggerCallback(experiment_name=experiment_location,save_artifact=False)],
                checkpoint_config = CheckpointConfig(num_to_keep = 1, 
                                                     checkpoint_score_attribute = 'eval_loss',
                                                     checkpoint_score_order = 'min') 
    ),
    datasets={"train": train_dataset , 
              "evaluation" : test_dataset},
    preprocessor=preprocessor,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we call the {meth}`~ray.train.huggingface.huggingface_trainer.HuggingFaceTrainer.fit` method to start training with Ray AIR. We will save the {class}`~ray.air.Result` object to a variable so we can access metrics and checkpoints.

# COMMAND ----------

# DBTITLE 0,glltddukjcuvujehr
results = trainer.fit()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch the best model parameters 

# COMMAND ----------

# MAGIC %md
# MAGIC You can use the returned {class}`~ray.air.Result` object to access metrics and the Ray AIR {class}`~ray.air.checkpoint.Checkpoint` associated with the last iteration.

# COMMAND ----------

checkpoint = results.checkpoint
checkpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate text from prompt

# COMMAND ----------

import ray.data
import pandas as pd
from training.generate import PredictCallable

instructions = [
    "Write a love letter to Edgar Allan Poe.",
    "Write a tweet announcing Dolly, a large language model from Databricks.",
    "I'm selling my Nikon D-750, write a short blurb for my ad.",
    "Explain to me the difference between nuclear fission and fusion.",
    "Give me a list of 5 science fiction books I should read next."]

ds = ray.data.from_pandas(pd.DataFrame(pd.Series(instructions),columns=["prompt"]))

# COMMAND ----------

preds = (
    ds
    .repartition(5)
    .map_batches(
        PredictCallable,
        batch_size=1,
        fn_constructor_kwargs=dict(checkpoint=checkpoint.uri.split('file://')[1],
                                   torch_dtype = "float16" ),#change to float16 when using V100
        batch_format="pandas",
        compute=ray.data.ActorPoolStrategy(),
        num_gpus=4,
    )
)
