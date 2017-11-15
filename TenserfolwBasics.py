import tensorflow as tf

# defining tensor as constant
node1 = tf.constant(6.0,dtype=tf.float32)
node2 = tf.constant(4.0)


#print the node
#print(node1,node2)

# Initializing session
sess = tf.Session()

#print the result as array of nodes
print(sess.run([node1,node2]))


# Adding the nodes
node3 = tf.add(node1,node2)


print('node3 :', node3)

print('session.run node3 :', sess.run(node3))


##################################################################


# Tensor Placeholders - hold type for now and assign latter

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = tf.add(a,b)


print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))


multy_node = tf.multiply(a,b)

print(sess.run(multy_node, {a:[3,5,7,9],b:[5,4,3,6]}))


div_node = tf.divide(a,b)

print(sess.run(div_node,{ a:[5,6,7,8,9],b:[6,7,8,9,2]}))

sub_node = tf.subtract(a,b)

print(sess.run(sub_node,{a:[45,78,89],b:[65,43,78]}))

all_node = tf.subtract(a,b)+tf.multiply(a,b)*tf.divide(a,b)

print(sess.run(all_node,{a:[34,56,78,89,54],b:[56,78,43,45,12]}))

# Tensor variables

W = tf.Variable([.3])
b = tf.Variable([-.3])

x = tf.placeholder(tf.float32)


linear_regration = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_regration, {x:[34,56,78,89,98]}))

# finding the lose

y = tf.placeholder(tf.float32)
squared_delta = tf.square(linear_regration - y)
loss = tf.reduce_sum(squared_delta)

print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))


fixb = tf.assign(W,[-1])
fixW = tf.assign(b, [1])

sess.run([fixb,fixW])

print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)

for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

print(sess.run([W,b]))

#Close the session
sess.close()