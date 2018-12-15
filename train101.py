import tensorflow as tf
import numpy as np

# random data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# tensorflow structure start

weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias = tf.Variable(tf.zeros([1]))

# prediction
y = weight * x_data + bias

# loss
loss = tf.reduce_mean(tf.square(y - y_data))
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
# train
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# tensroflow structure end

# init session
sess = tf.Session()
sess.run(init)

# run the train model to 200 times to adjust the weight and bias
for i in range(201):
	sess.run(train)
	if i % 20 == 0:
		print(i, sess.run(weight), sess.run(bias))

