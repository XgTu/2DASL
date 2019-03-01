def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
  """Decay exponentially in the later phase of training. All parameters in the 
  optimizer share the same learning rate.
  
  Args:
    optimizer: a pytorch `Optimizer` object
    base_lr: starting learning rate
    ep: current epoch, ep >= 1
    total_ep: total number of epochs to train
    start_decay_at_ep: start decaying at the BEGINNING of this epoch
  
  Example:
    base_lr = 2e-4
    total_ep = 300
    start_decay_at_ep = 201
    It means the learning rate starts at 2e-4 and begins decaying after 200 
    epochs. And training stops after 300 epochs.
  
  NOTE: 
    It is meant to be called at the BEGINNING of an epoch.
  """
  assert ep >= 1, "Current epoch number should be >= 1"

  if ep < start_decay_at_ep:
    return

  for g in optimizer.param_groups:
    g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                    / (total_ep + 1 - start_decay_at_ep))))
  print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))

'''
adjust_lr_exp(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.total_epochs,
        cfg.exp_decay_at_epoch)
'''