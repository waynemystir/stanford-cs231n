
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)
    x_hat = (x - sample_mean.T) / np.sqrt(sample_var.T + eps)

    out = x_hat * gamma + beta

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    cache = {}
    cache['sample_mean'] = sample_mean
    cache['sample_var'] = sample_var
    cache['x_hat'] = x_hat
    cache['x'] = x
    cache['gamma'] = gamma
    cache['beta'] = beta
    cache['eps'] = eps



    x_hat = (x - running_mean) / np.sqrt(running_var)
    out = x_hat * gamma + beta




  m = dout.shape[0]

  dx_hat = dout * cache['gamma'] 
  dsample_var = np.sum(dx_hat * (cache['x']-cache['sample_mean']) * (-0.5) * (cache['sample_var'] + cache['eps'])**(-1.5), axis=0)
  dsample_mean = np.sum(dx_hat * (-1/np.sqrt(cache['sample_var'] + cache['eps'])) , axis=0) + dsample_var * ((np.sum(-2*(cache['x']-cache['sample_mean']))) / m)

  dx = dx_hat * (1/np.sqrt(cache['sample_var'] + cache['eps'])) + \
       dsample_var * (2*(cache['x']-cache['sample_mean'])/m) + \
       dsample_mean/m

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout * cache['x_hat'], axis=0)
