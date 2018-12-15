import tensorflow as tf

# matrix 1 of 1 by 2
matrix1 = tf.constant([[3,3]])
# matrix 2 of 2 by 1
matrix2 = tf.constant([[2],
						[2]])

print(matrix1)
print(matrix2)

product = tf.matmul(matrix1, matrix2) # np.dot(m1,m2)

# way - 1 to run session
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# way - 2 to run session
with tf.Session() as sess:
	result2 = sess.run(product)
	print(result2)


