# Contrastive Predictive Coding for Natural Language

This repository contains a PyTorch implementation of CPC v1 for Natural Language (section 3.3) from the paper [Representation Learning with Contrastive Predictive Coding
](https://arxiv.org/abs/1807.03748).

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

**Extra details**
* batch of 64
* input is 6 sentences
* maximum sequence length of 32
* negative samples are drawn from both batch and time dimension in the minibatch
* uses Adam optimizer with a learning rate of 2e-4

## Usage Instructions



## Requirements


## References
* [Representation Learning with Contrastive Predictive Coding
](https://arxiv.org/abs/1807.03748)
* Part of the code is borrowed from https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch

