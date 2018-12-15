import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.div(input1,input2)

with tf.Session() as sess:
	result = sess.run(output, feed_dict={input1:[7.0], input2:[2.0]})
	print(result)


# same way as above

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

addition = tf.add(x,y)
substraction = tf.subtract(x,y)
multiplication = tf.multiply(x,y)
division = tf.divide(x,y)


sess = tf.Session()
print(addition, feed_dict={x:[5],y:[6]})
print(substraction, feed_dict={x:[10],y:[5]})
print(multiplication, feed_dict={x:[50],y:[5]})
print(division, fedd_dict={x:[50],y:[5]})