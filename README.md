# Cross-lingual Language Model Pretraining for Retrieval

Hi there! 👋 This repository contains code that supports experiments in our WWW 2021 paper ["Cross-lingual Language Model Pretraining for Retrieval"](https://dl.acm.org/doi/abs/10.1145/3442381.3449830). 


<p align="center">
  <img src="https://github.com/PxYu/Pretraining-CLIR/blob/master/pics/task.png" width=80% height=80%>  
</p>

Note that this is the PyTorch version of the implementation, which is largely based on the [XLM codebase](https://github.com/facebookresearch/XLM) by Facebook AI Research, and the Transformers library (**v3.4.0 recommended**) by HuggingFace. Many thanks to them! There is also a [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) version, which might be available upon request.

## What is this work doing?

Well, we observe that XLM and mBERT don't work well for multi-lingual **retrieval** tasks (ad-hoc retrieval, QA, etc). We propose to harvest the large amount of data from multi-lingual Wikipedia to strengthen Transformer-based models' ability to perform well on cross-lingual retrieval tasks. 

By first pretraining a multi-lingual language model on the supervision data with the QLM (query language modeling) and RR (relevance ranking) objectives, you can expect substantial improvement over mBERT and XLM on standard cross-lingual datasets, such as CLEF and MLQA.

<p align="center">
  <img src="https://github.com/PxYu/Pretraining-CLIR/blob/master/pics/perf.png" width=50% height=50%>
</p>

Even you are not interested in cross-lingual retrieval, you might still find our work helpful because our model can significantly increase semantic alignment of *long-sequence representations* (sentence and document level).

<p align="center">
  <img src="https://github.com/PxYu/Pretraining-CLIR/blob/master/pics/xda-perf.png" width=30% height=30%>
</p>

## How to use this repo?

Simple! There are just three steps to perform the experiments in our paper:

1. (optional) Convert a mBERT-base model into a BertLong model: `python BertLongForXLRetrieval.py`. This might help a lot if you are dealing with tasks that require long inputs, like document retrieval.
2. Pretraining with multi-lingual Wiki data: take a look at [train.py](https://github.com/PxYu/Pretraining-CLIR/blob/master/train.py)
3. Finetune and test: look at [finetune-search-5f.py](https://github.com/PxYu/Pretraining-CLIR/blob/master/finetune-search-5f.py) for CLIR on CLEF dataset; look at [finetune-qa.py](https://github.com/PxYu/Pretraining-CLIR/blob/master/finetune-qa.py) for CLQA on MLQA dataset.


### Some other tips:

- Shell scripts in `shell_scripts/` are some examples of how you should run those files in SLURM. Take a look if you run into trouble in this area!
- Data paths were hard-coded in the code, and unfortunately, I no longer have access to the environment where the model was developed. So I cannot change and test that. Sorry about that!🥺 But it should be very easy to fix for you.


### Resources

This [link](https://drive.google.com/drive/folders/1Ka4uRJqncxsQ5j7o8rgvRAAkE1IaaLV_?usp=sharing) contains the following resources:

- Processed multi-lingual Wiki for pretraining: Look at how these data were constructed in our paper!
- Pretrained checkpoints. Look at the first lines in the training logs to see how each model was trained.

I got asked about this a lot: I am **NOT** authorized to open-source processed CLEF evaluation data because it is not free. But if you have access to CLEF and have questions about how to run this model on the data, please contact me!


## Others

If you find our code or paper useful, please consider citing our work.

```
@inproceedings{yu2021cross,
  title={Cross-lingual Language Model Pretraining for Retrieval},
  author={Yu, Puxuan and Fei, Hongliang and Li, Ping},
  booktitle={Proceedings of the Web Conference 2021},
  pages={1029--1039},
  year={2021}
}
```

If you run into any problems regarding the code, open an issue! Contacting me at pxyu@cs.umass.edu is also encouraged.
