from __future__ import division

import tensorflow as tf
import numpy as np
import scipy.signal


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# KL divergence with itself, holding first argument fixed
def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd

    return gauss_KL(mu1, logstd1, mu2, logstd2)


# probability to take action x, given paramaterized guassian distribution
def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2 * logstd)
    gp = -tf.square(x - mu) / (2 * var) - .5 * tf.log(tf.constant(2 * np.pi)) - logstd
    return tf.reduce_sum(gp, [1])


# KL divergence between two paramaterized guassian distributions
def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2 * logstd1)
    var2 = tf.exp(2 * logstd2)

    kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2)) / (2 * var2) - 0.5)
    return kl


# Shannon entropy for a paramaterized guassian distributions
def gauss_ent(mu, logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32))
    return h


def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def cat_sample(prob_nk):
    assert prob_nk.ndim == 2
    # prob_nk: batchsize x actions
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype='i')
    for (n, csprob_k, r) in zip(range(N), csprob_nk, np.random.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out


def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


class FilterOb:
    def __init__(self, filter_mean=True):
        self.m1 = 0
        self.v = 0
        self.n = 0.
        self.filter_mean = filter_mean

    def __call__(self, obs):
        self.m1 = self.m1 * (self.n / (self.n + 1)) + obs * 1 / (1 + self.n)
        self.v = self.v * (self.n / (self.n + 1)) + (obs - self.m1) ** 2 * 1 / (1 + self.n)
        self.std = (self.v + 1e-6)**.5  # std
        self.n += 1
        if self.filter_mean:
            o1 = (obs - self.m1) / self.std
        else:
            o1 = obs / self.std
        o1 = (o1 > 10) * 10 + (o1 < -10) * (-10) + (o1 < 10) * (o1 > -10) * o1
        return o1


filter_ob = FilterOb()


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(axis=0, values=[tf.reshape(grad, [tf.size(v)]) for (v, grad) in zip(var_list, grads)])


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    # in numpy
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def make_network(network_name, input_layer, hidden_size, action_size):
    with tf.variable_scope(network_name):
        if len(input_layer.shape) > 2:
            size = np.prod(input_layer.get_shape().as_list()[1:])
            input_layer = tf.reshape(input_layer, [-1, size])

        h1, h1_vars = make_fully_connected("policy_h1", input_layer, hidden_size)
        h2, h2_vars = make_fully_connected("policy_h2", h1, hidden_size)
        h3, h3_vars = make_fully_connected("policy_h3", h2, action_size, final_op=None)

        logstd_action_dist_param = tf.Variable(
            (.01 * np.random.randn(1, action_size)).astype(np.float32), name="policy_logstd")

    variable_list = h1_vars + h2_vars + h3_vars + [logstd_action_dist_param]
    # means for each action
    avg_action_dist = h3
    # log standard deviations for each actions
    logstd_action_dist = tf.tile(logstd_action_dist_param, tf.stack((tf.shape(avg_action_dist)[0], 1)))

    return variable_list, avg_action_dist, logstd_action_dist


def make_fully_connected(
        layer_name,
        input_layer,
        output_size,
        final_op=tf.nn.relu,
        weight_init=tf.random_uniform_initializer(-0.05, 0.05),
        bias_init=tf.constant_initializer(0)):
    with tf.variable_scope(layer_name):
        w = tf.get_variable("w", [input_layer.shape[1], output_size], initializer=weight_init)
        b = tf.get_variable("b", [output_size], initializer=bias_init)
    raw = tf.matmul(input_layer, w) + b
    if final_op:
        return final_op(raw), [w, b]
    return raw, [w, b]
