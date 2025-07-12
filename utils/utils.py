def check_model_name(model_name):
    if model_name != "efficientdet-d0":
        raise ValueError("Only efficientdet-d0 is supported.")

def efficientdet_params(model_name):
    phi = 0
    w_bifpn = 64
    d_bifpn = 2
    d_class = 3
    return phi, w_bifpn, d_bifpn, d_class

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
