import tensorflow as tf

@tf.function
def VIN(X, S1, S2, args):
    
    k = args.k # Number of Value Iteration computations
    ch_i = args.ch_i # Channels in input layer
    ch_h = args.ch_h # Channels in initial hidden layer
    ch_q = args.ch_q # Channels in q layer (~actions)


    l_h = tf.keras.layers.Conv2D(filters=ch_h,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                                 bias_initializer=tf.zeros_initializer(),
                                 name='h0')
    h = l_h(X)
    # print('shape of h:', tf.shape(h))

    l_r = tf.keras.layers.Conv2D(filters=1,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                                 bias_initializer=None,
                                 name='h0')
    r = l_r(h)
    # print('shape of r:', tf.shape(r))

    # tf.Graph.add_to_collection(name='r', value=r)

    v = tf.zeros_like(r)
    rv = tf.concat([r, v], axis=3)
    # print('shape of rv:', tf.shape(rv))

    l_q = tf.keras.layers.Conv2D(filters=ch_q,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                                 bias_initializer=None,
                                 name='h0')
    q = l_q(rv)
    # print('shape of q:', tf.shape(q))
    v = tf.math.reduce_max(q, axis=3, keepdims=True, name='v')
    # print('shape of v:', tf.shape(h))

    for _ in range(k-1):
        rv = tf.concat([r, v], axis=3)
        q = l_q(rv)     # TODO: check whether neeed to create a new layer
        v = tf.math.reduce_max(q, axis=3, keepdims=True, name='v')

    # tf.Graph.add_to_collection(name='v', value=v)

    # one last convolution to get the q value
    rv = tf.concat([r, v], axis=3)
    q = l_q(rv)

    q_out = attention(q, S1, S2)

    # add the final fully connected layer
    fcl = tf.keras.layers.Dense(units=8,
                                activation=None,
                                use_bias=False,
                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                                name='logits')
    logits = fcl(q_out)
    # print('shape of logits:', tf.shape(logits))

    prob_actions = tf.nn.softmax(logits, name='probability_actions')
    
    return logits, prob_actions


@tf.function
def attention(tensor, S1, S2):

    s1 = tf.reshape(S1, [-1])
    s2 = tf.reshape(S2, [-1])

    N = tf.shape(tensor)[0]
    idx = tf.stack([tf.range(N), s1, s2], axis=1)

    q_out = tf.gather_nd(tensor, idx, name='q_out')
    
    return q_out