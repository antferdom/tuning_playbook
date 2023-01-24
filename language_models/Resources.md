# Resources

[TOC]

# Transformers

**Lectures:**

- [CS25: Transformers United V2](https://web.stanford.edu/class/cs25/): Since their introduction in 2017, transformers have revolutionized Natural Language Processing (NLP). Now, transformers are finding applications all over Deep Learning, be it computer vision (CV), reinforcement learning (RL), Generative Adversarial Networks (GANs), Speech or even Biology. Among other things, transformers have enabled the creation of powerful language models like GPT-3 and were instrumental in DeepMind's recent AlphaFold2, that tackles protein folding.

  In this seminar, we examine the details of how transformers work, and dive deep into the different kinds of transformers and how they're applied in different fields. We do this through a combination of instructor lectures, guest lectures, and classroom discussions. We will invite people at the forefront of transformers research across different domains for guest lectures.

  The bulk of this class will comprise of talks from researchers discussing latest breakthroughs with transformers and explaining how they apply them to their fields of research. The objective of the course is to bring together the ideas from ML, NLP, CV, biology and other communities on transformers, understand their broad implications, and spark cross-collaborative research.

## Inference Optimization

***ATTENTION:*** [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) (Lilian Weng)
- [Accelerating Text Generation with Confident Adaptive Language Modeling (CALM)](https://ai.googleblog.com/2022/12/accelerating-text-generation-with.html)

> _Presenting Confident Adaptive Language Modeling (CALM), a novel method that allows language models to dynamically modify computational effort when generating text. Learn how CALM can accelerate text generation while preserving output quality_

## Memory

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Paper:** https://arxiv.org/abs/2205.14135

**Lectures:** [Stanford’s Advances in Foundation Models Class (CS 324)](https://twitter.com/Avanika15/status/1612861499265142785)





## Context Window

### GPT Index

GPT Index is a project consisting of a set of data structures designed to make it easier to use large external knowledge bases with LLMs.

> _One of the downsides of text splitting is that it makes each text chunk lack global context. Now, thanks to some great work by [@thejessezhang](https://twitter.com/thejessezhang), we've made a big step in addressing this - injecting doc metadata in each chunk!_ [Jerry Liu](https://twitter.com/jerryjliu0)

**Github:** [jerryjliu](https://github.com/jerryjliu)/**[gpt_index](https://github.com/jerryjliu/gpt_index)**

# Fine-Tuning

1. [google-research](https://github.com/google-research)/**[tuning_playbook](https://github.com/google-research/tuning_playbook)**: A playbook for systematically maximizing the performance of deep learning models.

## Choosing the batch size

Continuously increasing batch size does not guarantee better training accuracy.

> _Regarding choosing a good batch size: Choosing a batch size to be **as large as your hardware permits** (up to a point) seems to be a **good** recommendation. Compiled some resources via the figure below._ [Sebastian Raschka](https://twitter.com/rasbt)

> _Once you have good estimate of gradient, **more examples don't help**. For least squares, largest convergent step size s=b/(x + y*b) where b is batch-size, x,y are constants. At some point, **increasing b doesn't affect s**, this gives lower bound on the number of steps needed._ [Yaroslav Bulatov](https://twitter.com/yaroslavvb)

## Loss-Function

- [A survey and taxonomy of loss functions in machine learning](https://ai.papers.bar/paper/f7292e3a954c11ed8f9c3d8021bca7c8)

> _Most state-of-the-art machine learning techniques revolve around the optimisation of loss functions. Defining appropriate loss functions is therefore critical to successfully solving problems in this field. We present a survey of the most commonly used loss functions for a wide range of different applications, divided into classification, regression, ranking, sample generation and energy based modelling. Overall, we introduce 33 different loss functions and we organise them into an intuitive taxonomy. Each loss function is given a theoretical backing and we describe where it is best used. This survey aims to provide a reference of the most essential loss functions for both beginner and advanced machine learning practitioners._ **(abstract)**


# Distillation
## Model Compression

**SetFit**

1. Quantization
2. Weights Prunning
3. Model distillation (teacher/student)

- [Cramming: Training a Language Model on Single GPU in ONE DAY](https://twitter.com/giffmana/status/1608568387583737856)

## Dataset

1. [Dataset Distillation: A Comprehensive Review](https://twitter.com/martin_gorner/status/1617464596201373696)

This field has attracted a surprising amount of research, yielding **4** broad sets of techniques:

- **Performance matching:** Distills a synthetic dataset "such that neural networks trained on it could have the lowest loss on the original dataset."
- **Parameter matching:** the network is trained in lockstep on real and synthetic data and the synthetic data is adjusted for the gradients in both networks to match.
- **Distribution matching:** Distilled data is synthesised directly to match the distribution of real data according to some metric. This can be done in the feature space.
- Data point selection, i.e. **pruning** the **dataset**.

### Synthethic Data

> ***LLMs can produce creative + correct data for training/eval of LLMs  -Faster & cheaper than humans -Humans agreed w/ "90-100% of labels, sometimes more so than corresponding human-written datasets.***
>
> [@AnthropicAI](https://twitter.com/AnthropicAI) & [@MetaAI](https://twitter.com/MetaAI)
>
>  papers: [https://arxiv.org/abs/2212.09689](https://t.co/K82BaFoH7l) [https://arxiv.org/abs/2212.09251](https://t.co/ZXxpPV6k14) [John Nay](https://twitter.com/johnjnay)



# Large Language Models Research

## Reasoning

- [Natural language as the most high level programming language](https://twitter.com/karpathy/status/1617566162199670784)

### Tools

- [TalkToCode](https://github.com/keerthanpg/TalkToCode)



## Reliability & Interpretability

### Main Sources

1. [Anthropic](https://twitter.com/AnthropicAI)

### Debugging & Reversing

- [Reverse engineering a neural network's clever solution to binary addition](https://cprimozic.net/blog/reverse-engineering-a-small-neural-network/)



## Online (Continuous) Learning

1. [olm-gpt2](https://twitter.com/joao_gante/status/1616468322069155840)



# Deep Learning Backend Theory

***ATTENTION:*** [A Call to Build Models Like We Build Open-Source Software](https://twitter.com/colinraffel/status/1468618801134592012) (Colin Raffel)

## Neural Network

### Optimizer

#### SGD

- **tinygrad impl:** [class SGD(Optimizer)](https://github.com/geohot/tinygrad/blob/7a159b9b047fffaec0339adf05bc9f08fb970b8d/tinygrad/nn/optim.py#LL28-L28C21)

#### Adam

- **tinygrad impl:** [class Adam(Optimizer)](https://github.com/geohot/tinygrad/blob/7a159b9b047fffaec0339adf05bc9f08fb970b8d/tinygrad/nn/optim.py#L51)

##### Learning Rate

> $3*10^{-4}$ is the best learning rate for Adam, hands down. [Andrej Karpathy](https://twitter.com/karpathy)

#### RMSprop

- **tinygrad impl:** [class RMSprop(Optimizer)](https://github.com/geohot/tinygrad/blob/7a159b9b047fffaec0339adf05bc9f08fb970b8d/tinygrad/nn/optim.py#L38)
- **PyTorch impl:** [torch.optim.RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop)







