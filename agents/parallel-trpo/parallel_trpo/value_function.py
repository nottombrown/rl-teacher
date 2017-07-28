import numpy as np
import tensorflow as tf
from parallel_trpo import utils

class VF(object):
    coeffs = None

    def __init__(self, session):
        self.net = None
        self.session = session

    def create_net(self, shape):
        hidden_size = 64
        self.x = tf.placeholder(tf.float32, shape=[None, shape], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")

        with tf.variable_scope("VF"):
            h1, _ = utils.make_fully_connected("h1", self.x, hidden_size)
            h2, _ = utils.make_fully_connected("h2", h1, hidden_size)
            h3, _ = utils.make_fully_connected("h3", h2, 1, final_op=None)
        self.net = tf.reshape(h3, (-1,))
        l2 = tf.nn.l2_loss(self.net - self.y)
        self.train = tf.train.AdamOptimizer().minimize(l2)
        self.session.run(tf.global_variables_initializer())

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        act = path["action_dists"].astype('float32')
        length = len(path["rewards"])
        al = np.arange(length).reshape(-1, 1) / 10.0
        ret = np.concatenate([o, act, al, np.ones((length, 1))], axis=1)
        return ret

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape[1])
        returns = np.concatenate([path["returns"] for path in paths])
        for _ in range(50):
            self.session.run(self.train, {self.x: featmat, self.y: returns})

    def predict(self, path):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            ret = self.session.run(self.net, {self.x: self._features(path)})
            return np.reshape(ret, (ret.shape[0],))

class LinearVF(object):
    coeffs = None

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        length = len(path["rewards"])
        al = np.arange(length).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, np.ones((length, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(featmat.T.dot(featmat) + lamb * np.identity(n_col), featmat.T.dot(returns))[0]

    def predict(self, path):
        if self.coeffs is None:
            return np.zeros(len(path["rewards"]))
        else:
            return self._features(path).dot(self.coeffs)
