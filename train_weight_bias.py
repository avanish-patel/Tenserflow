import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.2 + 3

# create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

# y predicted
y_predict = Weights * x_data + biases
# calculate the loss of y predicted and y actual
loss = tf.reduce_mean(tf.square(y_predict - y_data))
# optimizer to minimize the loss with learning rate of 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
#train to minimize the loss
train = optimizer.minimize(loss)
# initialize all variables
init = tf.initialize_all_variables()

# create tensorflow structure end
sess = tf.Session()
sess.run(init)


print("Step","  Weights","   Biases")

for i in range(201):
    # run session to train the network
    sess.run(train)
    # print weights and biases every 20 steps
    if i%20 == 0:
        print(i, sess.run(Weights), sess.run(biases))

# close the session
sess.close()




