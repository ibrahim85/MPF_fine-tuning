'''
This is a simple implementation of the adam algorithm.
'''

import numpy as np
import theano
import theano.tensor as T


def get_adam(grad, m_t = 0, v_t=0, epsilon=1e-8, lr = 0.01, beta_1_power=1, beta_2_power=1):

    beta1 = 0.9
    beta2 = 0.999
    m = beta1*m_t + (1-beta1)*grad
    v = beta2*v_t + (1-beta2)*grad*grad
    beta1_power = beta_1_power* beta1
    beta2_power = beta_2_power*beta2

    return m, v, beta1_power, beta2_power




def indi_adam(visible_units, hidden_units):
    a = np.ones((visible_units, hidden_units))
    b = np.zeros((visible_units,visible_units))
    c = np.zeros((hidden_units,hidden_units))
    zero_grad_u = np.concatenate((b,a),axis = 1)
    zero_grad_d = np.concatenate((a.T,c),axis=1)
    zero_grad = np.concatenate((zero_grad_u,zero_grad_d),axis=0)

    bias_grad = np.ones(hidden_units)

    return np.concatenate(zero_grad.ravel(),bias_grad.ravel())



def adam(grads, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf

    """
    updates = []
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

def Adam(grads, params, lr=0.0001, b1=0.9, b2=0.999, e=1e-8):
    updates = []
    i = theano.shared(np.float32(0))
    i_t = i + 1.
    fix1 = 1. - b1**i_t
    fix2 = 1. - b2**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = ((1. - b1) * g) + (b1 * m)
        v_t = ((1. - b2) * T.sqr(g)) + (b2 * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates