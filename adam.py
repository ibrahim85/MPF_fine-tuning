'''
This is a simple implementation of the adam algorithm.
'''

import numpy as np
def get_adam(grad, m_t = 0, v_t=0, epsilon=1e-8, lr = 0.01, beta_1_power=1, beta_2_power=1):

    beta1 = 0.9
    beta2 = 0.999
    m = beta1*m_t + (1-beta1)*grad
    v = beta2*v_t + (1-beta2)*grad*grad
    beta1_power = beta_1_power* beta1
    beta2_power = beta_2_power*beta2

    return m, v, beta1_power, beta2_power






