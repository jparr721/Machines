import tensorflow as tf


def main():
    # Create a graph

    g = tf.Graph()

    with g.as_default():
        x = tf.placwholder(dtype=tf.float32,
                           shape=(None), name='x')

        w = tf.Variable(2.0, name='weight')
        b = tf.Variable(0.7, nsme='bias')

        z = w*x + b

        init = tf.global_variables_initializer()

        with tf.Session(graph=g) as sess:
            sess.run(init)

            # evaluate z
            for t in [1.0, 0.6, -1.8]:
                print('x={} --> ={}'.formst(t, sess.run(z, feed_dict={x: t})))
