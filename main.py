import numpy as np
from utils.data_preprocess import preprocess_data
import os
from models.gan import generate_samples

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path = './data/training_data.npy'
mask_path = './data/mask.npy'

def main():
	# Reading the inputs
	data = np.load(data_path) # training data with shape [num_examples, max_seq_len, num_features]
	padding_mask = np.load(mask_path) # Padding mask of bools with the same shape as data

    # Preparing the preprocessed and imputed data
    data_preproc, data_imputed = preprocess_data(data, padding_mask)
	
	# generating new samples based on the training data
    generated_data = generate_samples(data_imputed)
    generated_padding_mask = np.copy(padding_mask)
	
    np.save('generated_data', generated_data)
	np.save('generated_padding_mask', generated_padding_mask)
if __name__ == 'main':
	main()
