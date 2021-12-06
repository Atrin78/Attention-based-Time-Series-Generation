# Transformer Guided Time-Series Generation

In this work we aim to generate high quality time-series data by capturing the dynamics of the training data using a transformer. The model consists of three main modules, namely the generator, the discriminator, and the transformer. The generator samples 
<img src="https://latex.codecogs.com/gif.latex?z" /> 
from the uniform distribution and maps it to 
<img src="https://latex.codecogs.com/gif.latex?\hat{x}" /> which is the desired generated time-series data.
In the generator architecture, a self-attention module is inserted in the middle of an multi-layer RNN to capture long-range dependencies in the data. On the other hand, the dicriminator which is a multi-layer RNN, tries to discriminate between the training data <img src="https://latex.codecogs.com/gif.latex?x" /> and the generated data <img src="https://latex.codecogs.com/gif.latex?\hat{x}" />. Finally, a transformer that is pre-trained on the training data is used to guide the generator. The transformer encoder takes a time-series and encodes its information which is then passed to the transformer decoder alongside with the encoder input time-series and a look-ahead mask. The task of the decoder is to predict the <img src="https://latex.codecogs.com/gif.latex?i" />th step of the input sequence given its preciding steps. So by pretraining the transformer on the training data it's forced to capture the dynamics of the training data which can be used to guide the training of the generator.

![block diagram](/images/TGTG.png)
*The block diagram of Transformer Guided Time-series Genration.*

The transformer is pre-trained using the MSE loss. The loss that is used for the training of the rest of the network consists of three terms. The first term forces the moments of the <img src="https://latex.codecogs.com/gif.latex?\hat{x}" /> batch to be the same as the <img src="https://latex.codecogs.com/gif.latex?x" /> batch. The second term is the WGAN-GP loss and the third is an MSE loss between <img src="https://latex.codecogs.com/gif.latex?\hat{x}" /> and its reconstruction by the transformer which is named <img src="https://latex.codecogs.com/gif.latex?x^*" />. Since the transformer is pre-trained with the training data, it forces <img src="https://latex.codecogs.com/gif.latex?\hat{x}" /> to exhibit the same dynamic behaviour as the real data. This loss is minimized by the generator and maximized by the discriminator (which actually only maximizes the WGAN-GP loss).

![loss](/images/loss.png)
*The GAN loss.*

## Running the code


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

