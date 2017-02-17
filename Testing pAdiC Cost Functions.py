  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  logits1 = tf.nn.softmax(logits)
  pAdic = (1/10)*(tf.py_func(vpAdicNorm, [logits1,tf_train_labels,7], tf.float32,stateful=True))
  loss = tf.reduce_mean(
   tf.nn.softmax_cross_entropy_with_logits(logits+pAdic, tf_train_labels))
  # Optimizer.
  optimizer = tf.train.AdadeltaOptimizer(learning_rate=100, rho=.95, epsilon=1e-07).minimize(loss)
