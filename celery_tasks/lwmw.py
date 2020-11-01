import numpy as np
import tensorflow as tf
import math
import timeit
from sklearn import datasets
import os

tf.compat.v1.disable_eager_execution()

def levmarq(settings, x_train, y_train, mu_init=3.0, min_error=1e-10, max_steps=100, mu_multiply=10, mu_divide=10, m_into_epoch=10, verbose=False):
    outs = settings["outs"]
    m = settings["input_len"]
    print(5 * "=" + ">Training info<" + 5 * "=", "\n")
    print("Settings: ")
    for i in settings.keys():
        print(f"         {i}:{settings[i]}")
    print("\ntf version: ", tf.__version__, "\n")
    print(f"shape X:\t{x_train.shape}\nshape y:\t{y_train.shape}\n      m:\t{m}\n      p:\t{outs}")
    print("\n")

    x = tf.compat.v1.placeholder(tf.float64, shape=[m, settings["inputs"]])
    y = tf.compat.v1.placeholder(tf.float64, shape=[m, settings["outs"]])

    # hidden layers
    nn = settings["architecture"]

    st = [x_train.shape[-1]]+nn+[y_train.shape[-1]]

    sizes = []
    shapes = []
    for i in range(len(nn)+1):
        shapes.append((st[i], st[i+1]))
        shapes.append((1, st[i+1]))
    sizes = [h*w for h, w in shapes]
    neurons_cnt = sum(sizes)
    
    print(f"Complex:\n        [parameters]x[data lenth]\n        {neurons_cnt}x{m}\n")

    if settings["activation"] == "relu":
        activation = tf.nn.relu
    if settings["activation"] == "tanh":
        activation = tf.nn.tanh
    else:
        activation = tf.nn.sigmoid

    # feed forward
    initializer = tf.contrib.layers.xavier_initializer()
    p = tf.Variable(initializer([neurons_cnt], dtype=tf.float64))
    parms = tf.split(p, sizes, 0)
    for i in range(len(parms)):
        parms[i] = tf.reshape(parms[i], shapes[i])
    Ws = parms[0:][::2]
    bs = parms[1:][::2]

    y_hat = x
    for i in range(len(nn)):
        y_hat = activation(tf.matmul(y_hat, Ws[i]) + bs[i])
    y_hat = tf.matmul(y_hat, Ws[-1])+bs[-1]
    y_hat_flat = tf.squeeze(y_hat)

    r = y - y_hat
    loss = tf.reduce_mean(tf.square(r))

    # feed dicts for map placeholders to actual values

    train_dict = {
        x : x_train,
        y : y_train
    }

    Error_estimate = 10 * math.log10(1/(4*len(x_train) * int(y_train.shape[-1])))

    opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1)

    mu = tf.compat.v1.placeholder(tf.float64, shape=[1])

    p_store = tf.Variable(tf.zeros([neurons_cnt], dtype=tf.float64))

    save_parms =  tf.compat.v1.assign(p_store, p)
    restore_parms =  tf.compat.v1.assign(p, p_store)


    def jacobian(y, x, m):
        loop_vars = [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float64, size=m),
        ]

        _, jacobian = tf.while_loop(
            lambda i, _: i < m,
            lambda i, res: (i+1, res.write(i, tf.gradients(y[i], x)[0])),
            loop_vars)

        return jacobian.stack()

    I = tf.eye(neurons_cnt, dtype=tf.float64)

    j = jacobian(y_hat_flat, p, m)
    jT = tf.transpose(j)
    jTj = tf.matmul(jT, j)
    jTr = tf.matmul(jT, r)
    jTj = tf.hessians(loss, p)[0]
    jTr = -tf.gradients(loss, p)[0]
    jTr = tf.reshape(jTr, shape=(neurons_cnt, 1))

    jTj_store = tf.Variable(tf.zeros((neurons_cnt, neurons_cnt), dtype=tf.float64))
    jTr_store = tf.Variable(tf.zeros((neurons_cnt, 1), dtype=tf.float64))
    save_jTj_jTr = [ tf.compat.v1.assign(jTj_store, jTj),  tf.compat.v1.assign(jTr_store, jTr)]

    dx = tf.matmul(tf.linalg.inv(jTj_store + tf.multiply(mu, I)), jTr_store)
    dx = tf.squeeze(dx)
    _dx = tf.matmul(tf.linalg.inv(jTj + tf.multiply(mu, I)), jTr)
    _dx = -tf.squeeze(_dx)

    lm = opt.apply_gradients([(-dx, p)])

    # Train
    session = tf.compat.v1.Session()
    
    train_dict[mu] = np.array([mu_init])
    history = []
    step = 0
    session.run(tf.compat.v1.global_variables_initializer())
    current_loss = session.run(loss, train_dict)
    while current_loss > min_error and step < max_steps:
        step += 1
        if step % int(max_steps / 5) == 0 and verbose:
            print(f'LM step: {step}, mu: {train_dict[mu][0]:.2e}, current loss: {current_loss:.2e}')
        session.run(save_parms)
        session.run(save_jTj_jTr, train_dict)
        success = False
        for i in range(m_into_epoch):
            session.run(lm, train_dict)
            new_loss = session.run(loss, train_dict)
            if new_loss < current_loss:
                train_dict[mu] /= mu_divide
                current_loss = new_loss
                success = True
                break
            train_dict[mu] *= mu_multiply
            session.run(restore_parms)
        history.append(current_loss)
        if not success:
            print(f'LM failed to improve, on step {step:}, loss: {current_loss:.2e}\n')
            tp = session.run(p)
            session.close()
            tf.compat.v1.reset_default_graph()
            return np.asarray(history), tp
            break   

    print(f'LevMarq ended on: {step:},\tfinal loss: {current_loss:.2e}\n')
    tp = session.run(p)
    session.close()
    tf.compat.v1.reset_default_graph()
    return np.asarray(history), tp

def predict(p_input, settings, x_pred):
    outs = settings["outs"]
    m = len(x_pred)

    x = tf.compat.v1.placeholder(tf.float64, shape=[m, settings["inputs"]])

    # hidden layers
    nn = settings["architecture"]

    st = [x_pred.shape[-1]]+nn+[outs]

    sizes = []
    shapes = []
    for i in range(len(nn)+1):
        shapes.append((st[i], st[i+1]))
        shapes.append((1, st[i+1]))
    sizes = [h*w for h, w in shapes]
    neurons_cnt = sum(sizes)

    if settings["activation"] == "relu":
        activation = tf.nn.relu
    if settings["activation"] == "tanh":
        activation = tf.nn.tanh
    else:
        activation = tf.nn.sigmoid

    # feed forward
    p = tf.Variable(tf.constant(p_input), dtype=tf.float64)
    parms = tf.split(p, sizes, 0)
    for i in range(len(parms)):
        parms[i] = tf.reshape(parms[i], shapes[i])
    Ws = parms[0:][::2]
    bs = parms[1:][::2]

    y_hat = x
    for i in range(len(nn)):
        y_hat = activation(tf.matmul(y_hat, Ws[i]) + bs[i])
    y_hat = tf.matmul(y_hat, Ws[-1])+bs[-1]
    
    grads = tf.compat.v1.gradients(y_hat, x)
    
    session = tf.compat.v1.Session()
    session.run(tf.compat.v1.global_variables_initializer())
    ty = session.run(y_hat, {x: x_pred})
    grads_np = session.run(grads, {x: x_pred})
    session.close()
    tf.compat.v1.reset_default_graph()
    return ty, grads_np
