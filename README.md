# Contrastive Predictive Coding for Natural Language

This repository contains a PyTorch implementation of CPC v1 for Natural Language (section 3.3) from the paper [Representation Learning with Contrastive Predictive Coding
](https://arxiv.org/abs/1807.03748).

<p align="center"> 
    <img src="images/cpc-model.png" width="600">
</p>

## Implementation Details

I followed the details mentioned in section 3.3. Also, I got missing details directly from one of the paper's authors.

**Embedding layer**
* vocabulary size: 20 000
* dimension: 620

**Encoder layer (g_enc)**
* 1D-convolution + ReLU + mean-pooling
* output dimension: 2400

**Recurrent Layer (g_ar)**
* GRU
* dimension: 2400

**Prediction Layer {W_k}**
* Fully connected
* timesteps: 3

**Training details**
* input is 6 sentences
* maximum sequence length of 32
* negative samples are drawn from both batch and time dimension in the minibatch
* uses Adam optimizer with a learning rate of 2e-4
* trained on 8 GPUs, each with a batch size of 64

## Requirements

## Usage Instructions

### 1. Pretraining

**Configuration File**

This implementation uses a configuration file for convenient configuration of the model. The `config_cpc.yaml` file includes original parameters by default.
You have to adjust the following parameters to get started:
* `logging_dir`: directory for logging files
* `books_path`: directory containing the dataset

Optionally, if you want to log your experiments with [comet.ml](https://www.comet.ml/), you just need to install the library and write your `api_key`.

**Dataset**

This model uses [BookCorpus](http://yknzhu.wixsite.com/mbweb) dataset for pretrainig. You have to organize your data according to the following structure:
```
├── BookCorpus
│   └── data
│       ├── file_1.txt
│       ├── file_2.txt 
```
Then you have to write the path of your dataset in the `books_path` parameter of the `config_cpc.yaml` file.

*Note: You could use publicly available files provided by [Igor Brigadir](https://twitter.com/IgorBrigadir/status/1095075607178870786) at your own risk.*

**Training**

When you have completed all the steps above, you can run:

``python main.py``

The implementation automatically saves a log of the experiment with the name `cpc-date-hour` and also saves the model checkpoints with the same name.

**Resume Training**

If you want to resume your model training, you just need to write the name of your experiment (`cpc-date-hour`) in the `resume_name` parameter of the `config_cpc.yaml` file and then run `train.py`.

### 2. Vocabulary Expansion

The CPC model employs vocabulary expansion in the same way as the [Skip-Thought  model](https://arxiv.org/abs/1506.06726). You just need to modify the `run_name` and `word2vec_path` parameters to then execute:

``python vocab_expansion.py``

The result is a numpy file of embeddings and a pickle file of the vocabulary. They will appear in a folder named `vocab_expansion/`.

### 3. Training a Classifier

**Configuration File**

This implementation uses a configuration file for configuration of the classfier. You have to set the following parameters of the `config_clf.yaml` file:
* `logging_dir`: directory for logging files
* `cpc_path`: path of the pretrained cpc model file
* `expanded_vocab`: `True` if you want to use expanded vocabulary
* `dataset_path`: directory containing all the benchmark
* `dataset_name`: name of the task (e.g. CR, TREC, etc.)

**Dataset**

This classifier uses a common NLP benchmark. You have to organize your data according to the following structure:
```
├── dataset_name
│   └── data
│       └── task_name
│           ├── task_name.train.txt
│           ├── task_name.dev.txt 
```
Then you have to set the path of your data (`dataset_path`) and task name (`dataset_name`) in the `config_cpc.yaml` file.

*Note: You could use publicly available files provided by [zenRRan](https://github.com/zenRRan/Sentiment-Analysis/tree/master/data).*

**Training**

When you have completed the steps above, you can run:

``python main_clf.py``

The implementation automatically saves a log of the experiment with the name `cpc-clf-date-hour` and also saves the model checkpoints with the same name.

## Disclaimer

The model should be trained for 1e8 steps with a batch size of 64 * 8 GPUs. The authors provided me a snapshot of the first 1M training steps that you can find [here](https://github.com/vgaraujov/CPC-NLP-PyTorch/raw/master/images/deepmind-train-plot.png), and you can find the results of my implementation [here](https://github.com/vgaraujov/CPC-NLP-PyTorch/raw/master/images/varaujov-trian-plot.jpg). There is a slight difference which may be due to various factors such as dataset or initialization. I have not been able to train the model entirely, so I did not replicate the results with the benchmark.

If anyone can fully train the model, feel free to share the results. I will be attentive to any questions or comments. 

## References
* [Representation Learning with Contrastive Predictive Coding
](https://arxiv.org/abs/1807.03748)
* Part of the code is borrowed from https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch
* Part of the code is borrowed from https://github.com/ryankiros/skip-thoughts

