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

First, put your training data alongside with its padding mask in the data folder. The data path settings can be modified in the main.py file. The synthetic data will be generated using the code below.

```
python3 main.py
```


