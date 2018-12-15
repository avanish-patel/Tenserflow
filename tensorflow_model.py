import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define function that add layer in NN
def add_layer(inputs, in_size, out_size, activation_function=None):

    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    
    return outputs


# make up some real data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


# print(x_data)
# plt.scatter(x_data,y_data)
# plt.show()

# define placeholders for network
xs = tf.placeholder(tf.float32, [None,1])
ys = tf.placeholder(tf.float32, [None,1])

# add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1,10,1, activation_function= None)

# error between real and prediction
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()


# run session
sess = tf.Session()

sess.run(init)

for i in range(2000):
    #training
    sess.run(train, feed_dict={xs:x_data, ys:y_data})
    
    if i % 100 == 0:
        # print steps
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))