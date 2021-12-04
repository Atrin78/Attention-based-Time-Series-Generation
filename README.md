# Transformer Guided Time-Series Generation

In this work we aim to generate high quality time-series data by capturing the dynamics of the training data using a transformer. The model consists of three main modules, namely the generator, the discriminator, and the transformer. The generator drawes a sample named 
<img src="https://latex.codecogs.com/gif.latex?z" /> 
from the uniform distribution and mapps it to 
<img src="https://latex.codecogs.com/gif.latex?\hat{x}" /> 
For the generator, a self-attention module is inserted in the middle of an multi-layer RNN to capture long-range dependencies in the  data.

![ARAE vs DAE](/MNIST-union/images/ARAEvsDAE.png)
*Unlike  DAE,  ARAE  that  is  trained  on  the  normal  class,  which  is  the digit 8, reconstructs a normal instance when it is given an anomalous digit, from the class 1.*

<!--
Here, we can provide the link to our paper, and we can write authors list.

<!--
This repository belongs to abnormal detection group in Sharif university of Technology. This project is under supervision of [Dr. Mohammad Hossein Rohban](https://scholar.google.com/citations?user=pRyJ6FkAAAAJ&hl=en) and is being conducted in [Data Science and Machine Learning Lab (DML)](http://dml.ir/) in Department of Computer Engineering. -->

<!--
The aim of the project is to learn a robust representation from normal samples in order to detect abnormality patterns. This work is mainly inspired by these papers, ["Adversarial examples for generative models"](https://arxiv.org/pdf/1702.06832.pdf) and ["Adversarial Manipulation of Deep Representations"](https://arxiv.org/pdf/1511.05122.pdf). More specifically, a new objective function is introduced by which an Autoencoder is trained so that it can both minimize pixel-wise error and learn a robust representation where it can capture variants of a sample in latesnt space. -->

## Running the code![68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6861747b787d](https://user-images.githubusercontent.com/33551918/144698702-abbf2790-8c1a-475e-b8d9-672e2e2d3841.gif)


Having cloned the repository, you can reproduce our results:

### 1. L-inf model:

If you want to use the pre-trained models, skip to [this section](https://github.com/rohban-lab/Salehi_submitted_2020#testing).

#### Preparing the data

At first, run prepare.py to prepare the data. The first argument to be passed is the dataset name. You may choose between fashion_mnist, mnist, and coil100.  For mnist and fashion_mnist, the next argument is the chosen protocol to prepare the data. For this argument, you may choose between p1 and p2. If p2 is chosen, the next argument is the normal class number. Otherwise, the next argument is the anomaly percentage. Then you have to pass the class number.

Here are two examples for mnist and fashion_mnist datasets:

```
python3 prepare.py mnist p1 0.5 8
```
```
python3 prepare.py fashion_mnist p2 2
```

