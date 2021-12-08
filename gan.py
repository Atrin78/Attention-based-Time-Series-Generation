
import tensorflow as tf
import numpy as np
from math import ceil
import time
from .utils import ( 
                    extract_time,
                    rnn_cell,
                    random_generator,
                    batch_generator,
                    TokenAndPositionEmbedding,
                    EncoderLayer,
                    DecoderLayer,
                    create_look_ahead_mask,
                    CustomSchedule,
                    )



def generate_samples(ori_data):
  """Generation function.
  
  Use original data as training set to generate synthetic data (time-series)
  
  Args:
    - ori_data: original time-series data
    - parameters: GAN and transformer networks parameters
    
  Returns:
    - generated_data: generated time-series data
  """
  start = time.time()
  parameters = dict()  
  parameters['module'] = 'gru'
  parameters['hidden_dim'] = 24
  parameters['num_layer'] = 3
  parameters['iterations'] = 20000
  parameters['batch_size'] = 128
  parameters['d_model'] = 24
  parameters['num_heads'] = 2
  parameters['dff'] = 128
  
  # Initialization on the Graph
  tf.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Maximum sequence length and each sequence length
  ori_time, max_seq_len = extract_time(ori_data)
  
  def MinMaxScaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
      
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val
  
  # Normalization
  ori_data, min_val, max_val = MinMaxScaler(ori_data)
  print(ori_data.shape)
              
  ## Building The GAN and transformer networks
  
  # Network Parameters
  # The num_layers and gamma parameters are not used
  hidden_dim   = parameters['hidden_dim']
  num_layers   = parameters['num_layer']
  iterations   = parameters['iterations']
  batch_size   = parameters['batch_size']
  module_name  = parameters['module']
  d_model = parameters['d_model']
  num_heads = parameters['num_heads']
  dff = parameters['dff']
  z_dim        = dim
  gamma        = 1
  # look ahead mask for the transformer decoder
  mask = create_look_ahead_mask(max_seq_len)
    
  # Input place holders
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name = "myinput_z")
  T = tf.placeholder(tf.int32, [None], name = "myinput_t")
  training = tf.placeholder(tf.bool, shape=())


  
  def embedder (X, T):
    """Embedding network between original feature space to latent space. This is the embedding module of the transformer.
    
    Args:
      - X: input time-series features
      - T: input time information
      
    Returns:
      - H: embeddings
    """
    with tf.variable_scope("embedder", reuse = tf.AUTO_REUSE):
      # layer to embed the position information
      embedding_layer = TokenAndPositionEmbedding(max_seq_len, d_model, dff, True)
      x = embedding_layer(X)
      # Self attention layers
      encoder_block1 = EncoderLayer(d_model, num_heads, dff)
      encoder_block2 = EncoderLayer(d_model, num_heads, dff)
      encoder_block3 = EncoderLayer(d_model, num_heads, dff)
      
      x = encoder_block1(x, training, None)
      x = encoder_block2(x, training, None)
      H = encoder_block3(x, training, None)
    
    return H
      
  def recovery (X1, H1):
    """Recovery network from latent space to original space. This is the decoder module of the transformer.
    
    Args:
      - H1: latent representation
      - X1: input data
      
    Returns:
      - X_tilde1: recovered data
    """     
    with tf.variable_scope("recovery", reuse = tf.AUTO_REUSE):
      # layer to embed the position information
      embedding_layer = TokenAndPositionEmbedding(max_seq_len, d_model, dff, True)
      # multi-head attention layers
      decoder_block1 = DecoderLayer(d_model, num_heads, dff)
      decoder_block2 = DecoderLayer(d_model, num_heads, dff)
      decoder_block3 = DecoderLayer(d_model, num_heads, dff)
      final = tf.keras.layers.Dense(dim)
      
      
      x = tf.concat([tf.zeros_like(X1[:, :1, :]), X1[:, :-1, :]], axis=1)
      x = embedding_layer(x)
      x = decoder_block1(x, H1, training, mask, None)
      x = decoder_block2(x, H1, training, mask, None)
      x = decoder_block3(x, H1, training, mask, None)
      
      X_tilde1 = final(x)

    return X_tilde1
    


  # encoding the original data using the transformer embedder
  H = embedder(X, T)

  # Generating the synthetic time-series data
  X_hat = supervisor(generator(Z, T), T)
  # encoding the synthetic data using the transformer embedder
  E_hat = embedder(X_hat, T)
  
  # decoding using the transformer decoder
  X_tilde = recovery(X, H)
  X_hat_e = recovery(X_hat, E_hat)
  # Discriminator
  Y_real = discriminator(X, T)
  Y_fake = discriminator(X_hat, T)
    
  # Variables        
  e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
  g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
  s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    
  # Discriminator WGAN-GP loss
  # 1. real data loss
  D_loss_real = tf.math.reduce_mean(Y_real)
  # 2. fake data loss
  D_loss_fake = tf.math.reduce_mean(Y_fake)
  # 3. gradient penalty
  alpha = tf.random.uniform(
                            shape=[batch_size, 1, 1],
                            minval=0.,
                            maxval=1.
                            )
  real_data = X
  fake_data = X_hat
  differences = fake_data - real_data
  interpolates = real_data + (alpha*differences)
  gradients = tf.gradients(tf.math.reduce_mean(discriminator(interpolates, T), axis=[1]), [interpolates])[0]
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
  gradient_penalty = tf.reduce_mean((slopes-1.)**2)
  # 4. summation
  D_loss = -D_loss_real + D_loss_fake + 10 * gradient_penalty
  
            
  # Generator loss
  # 1. Adversarial loss (WGAN-GP)
  G_loss_U = -tf.math.reduce_mean(Y_fake)

  # 2. Supervised loss
  G_loss_S = tf.losses.mean_squared_error(X_hat, X_hat_e)

  # 3. Two Momments
  G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
  G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    
  G_loss_V = G_loss_V1 + G_loss_V2
    
  # 4. Summation
  G_loss = G_loss_U + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
            
  # transformer network loss for pretraining
  E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
  E_loss0 = 10 * E_loss_T0

  # optimizer for pretraining the transformer
  E0_solver = tf.train.AdamOptimizer(
                                     learning_rate=1e-4,
                                     beta1=0,
                                     beta2=0.9
                                    ).minimize(E_loss0, var_list = e_vars + r_vars)
  # optimizer for training the discriminator
  D_solver = tf.train.AdamOptimizer(
                                    learning_rate=1e-4,
                                    beta1=0,
                                    beta2=0.9
                                    ).minimize(D_loss, var_list = d_vars)
  # optimizer for training the generator
  G_solver = tf.train.AdamOptimizer(
                                       learning_rate=1e-4,
                                       beta1=0,
                                       beta2=0.9
                                       ).minimize(G_loss, var_list = g_vars + s_vars)


  ## training
  # initializing weights
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())

  # 1. transformer network pre-training
  print('Start Embedding Network Training')
  for itt in range(iterations):
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
    # Train transformer
    _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb, training:True})
    # Checkpoint
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(step_e_loss,4)) )

  print('Finish Embedding Network Training')


  # 2. GAN Training
  print('Start Joint Training')
  for itt in range(iterations):
    if itt % 100 == 0:
      print('step: '+ str(itt) + '/' + str(iterations))
    
    # train the generator two times
    for kk in range(2):
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      # Train generator
      _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb, training:True})
      
      if itt % 100 == 0:
        print(', g_loss_u: ' + str(np.round(step_g_loss_u,4)) +
              ', g_loss_s: ' + str(np.round(step_g_loss_s,4)) +
              ', g_loss_v: ' + str(np.round(step_g_loss_v,4))  )

    # Discriminator training
    for kk in range(1):
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      # Train discriminator
      _, step_d_loss, step_d_loss_real, step_d_loss_fake, gp = sess.run([D_solver, D_loss, D_loss_real, D_loss_fake, gradient_penalty], feed_dict={X: X_mb, T: T_mb, Z: Z_mb, training:True})

    # Print multiple checkpoints
      if itt % 100 == 0:
        print(', d_loss: ' + str(np.round(step_d_loss,4)) +
            ', d_loss_real: ' + str(np.round(step_d_loss_real,4)) +
            ', d_loss_fake: ' + str(np.round(step_d_loss_fake,4)) +
            ', gp: ' + str(np.round(gp,4)))
  print('Finish Joint Training')

  ## Synthetic data generation
  num_batches = int(ceil(no / batch_size))
  generated_data_curr = None
  for i in range(num_batches):
    if i != num_batches-1:
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, ori_time, max_seq_len)
      if generated_data_curr is None:
        # generating sythetic data for this batch
        generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data[i*batch_size:(i+1)*batch_size], T: ori_time[i*batch_size:(i+1)*batch_size], training:False})
      else:
        # generating sythetic data for this batch
        generated_data_curr = np.concatenate((generated_data_curr, sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data[i*batch_size:(i+1)*batch_size], T: ori_time[i*batch_size:(i+1)*batch_size], training:False})), axis=0)
    else:
      # Random vector generation
      Z_mb = random_generator(no-i*batch_size, z_dim, ori_time, max_seq_len)
      if generated_data_curr is None:
        # generating sythetic data for this batch
        generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data[i*batch_size:], T: ori_time[i*batch_size:], training:False})
      else:
        # generating sythetic data for this batch
        generated_data_curr = np.concatenate((generated_data_curr, sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data[i*batch_size:], T: ori_time[i*batch_size:], training:False})), axis=0)
    
  generated_data = list()
    
  for i in range(no):
    temp = generated_data_curr[i,:ori_time[i],:]
    generated_data.append(temp)
        
  # Renormalization
  generated_data = generated_data * max_val
  generated_data = generated_data + min_val

  print('overall time')
  print(time.time() - start)

  return generated_data
