# Codebase for WWW 2021 paper "Cross-lingual Language Model Pretraining for Retrieval"

This repository contains code that supports experiments in our WWW 2021 paper [Cross-lingual Language Model Pretraining for Retrieval](https://dl.acm.org/doi/abs/10.1145/3442381.3449830). Note that this is the PyTorch version of the implementation, which is largely based on the [XLM codebase](https://github.com/facebookresearch/XLM) by Facebook AI Research, and the Transformers library by HuggingFace. Many thanks to them! 

There is also a [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) version. It might be available upon request. 



to convert a two-way Bert model based on multilingual-bert to a two-way BertLong model, run

```
python BertLongForXLRetrieval.py
```