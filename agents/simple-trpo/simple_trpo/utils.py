from __future__ import division

import random

import numpy as np
import scipy.signal
import tensorflow as tf

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def gauss_prob(mu, logstd, x):
    std = tf.exp(logstd)
    var = tf.square(std)
    gp = tf.exp(-(x - mu) / (2 * var)) / ((2 * np.pi) ** .5 * std)
    return tf.reduce_prod(gp, [1])

def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2 * logstd)
    gp = -tf.square(x - mu) / (2 * var) - .5 * tf.log(tf.constant(2 * np.pi)) - logstd
    return tf.reduce_sum(gp, [1])

def gauss_selfKL_firstfixed(mu, logstd):
    """Compute KL with yourself, where the first argument is treated as a constant. """
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd

    return gauss_KL(mu1, logstd1, mu2, logstd2)

def gauss_KL(mu1, logstd1, mu2, logstd2):
    """Compute KL"""
    var1 = tf.exp(2 * logstd1)
    var2 = tf.exp(2 * logstd2)

    kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2)) / (2 * var2) - 0.5)
    return kl

def gauss_ent(mu, logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32))
    return h

def gauss_sample(mu, logstd):
    return mu + tf.exp(logstd) * tf.random_normal(tf.shape(logstd))

class ObsFilter:
    def __init__(self, filter_mean=True):
        self.m1 = 0
        self.v = 0
        self.n = 0.
        self.filter_mean = filter_mean

    def __call__(self, o):
        self.m1 = self.m1 * (self.n / (self.n + 1)) + o * 1 / (1 + self.n)
        self.v = self.v * (self.n / (self.n + 1)) + (o - self.m1) ** 2 * 1 / (1 + self.n)
        self.std = (self.v + 1e-6) ** .5  # std
        self.n += 1
        if self.filter_mean:
            o1 = (o - self.m1) / self.std
        else:
            o1 = o / self.std
        o1 = (o1 > 10) * 10 + (o1 < -10) * (-10) + (o1 < 10) * (o1 > -10) * o1
        return o1

obs_filter = ObsFilter()


def vectorized_rollout(env, agent, max_timesteps_per_episode, timesteps_per_batch, render=False):
    paths = []
    timesteps_elapsed = 0
    first = True
    while timesteps_elapsed < timesteps_per_batch:
        obs, actions, rewards, action_dists_mu, action_dists_logstd = [], [], [], [], []
        ob = obs_filter(env.reset())
        for _ in range(max_timesteps_per_episode):
            timesteps_elapsed += 1
            obs.append(ob)
            action, info = agent.act(ob)
            actions.append(action)
            action_dists_mu.append(info.get("action_dist_mu", []))
            action_dists_logstd.append(info.get("action_dist_logstd", []))
            ob, reward, done, info = env.step(action)
            ob = obs_filter(ob)
            rewards.append(reward)
            if render and first: env.render()
            if done or timesteps_elapsed == timesteps_per_batch:
                # forceful termination if timesteps_sofar == n_timesteps
                # otherwise paths is empty, which also is bad.
                path = ConfigObject(obs=np.concatenate(np.expand_dims(obs, 0)),
                    action_dists_mu=np.concatenate(action_dists_mu),
                    action_dists_logstd=np.concatenate(action_dists_logstd),
                    rewards=np.array(rewards),
                    actions=np.array(actions))
                paths.append(path)
                # print("steps in ep: ", len(rewards))
                # print("maxsteps in ep: ", max_timesteps_per_episode)
                break

        first = False
    return paths

class LinearVF(object):
    coeffs = None

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(featmat.T.dot(featmat) + lamb * np.identity(n_col), featmat.T.dot(returns))[0]

    def predict(self, path):
        if self.coeffs is None:
            return np.zeros(len(path["rewards"]))

        return self._features(path).dot(self.coeffs)

def cat_sample(prob_nk):
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype='i')
    for (n, csprob_k, r) in zip(range(N), csprob_nk, np.random.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out),\
        "shape function assumes that shape is fully known"
    return out

def num_els(x):
    return np.prod(var_shape(x))

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(axis=0, values=[tf.reshape(grad, [num_els(v)]) for (v, grad) in zip(var_list, grads)])

class SetFromFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        shapes = list(map(var_shape, var_list))  # note, here is the needed change.
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(tf.assign(v, tf.reshape(self.theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})

class GetFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [num_els(v)]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)

def slice_2d(x, inds0, inds1):
    # in tf
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)

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

class ConfigObject(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self
