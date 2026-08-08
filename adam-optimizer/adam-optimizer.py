import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    
    mt = beta1 * m + (1.0 - beta1) * grad
    vt = beta2 * v + (1.0 - beta2) * (grad)**2

    mt_cor = mt / (1.0 - (beta1)**t)
    vt_cor = vt / (1.0 - (beta2)**t)

    param_new = param - lr * (mt_cor/(np.sqrt(vt_cor) + eps))

    return param_new, mt, vt
    