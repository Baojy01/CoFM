from .cofm import CoFM_Micro, CoFM_Miny, CoFM_Tiny, CoFM_Small, CoFM_Base, CoFM_Large

models = ['CoFM_Micro','CoFM_Miny', 'CoFM_Tiny', 'CoFM_Small', 'CoFM_Base', 'CoFM_Large']         


def get_model(name, num_class):

    if name.lower() == 'cofm_micro':
        net = CoFM_Micro(num_class)
    elif name.lower() == 'cofm_miny':
        net = CoFM_Miny(num_class)
    elif name.lower() == 'cofm_tiny':
        net = CoFM_Tiny(num_class)
    elif name.lower() == 'cofm_small':
        net = CoFM_Small(num_class)
    elif name.lower() == 'cofm_base':
        net = CoFM_Base(num_class)
    elif name.lower() == 'cofm_large':
        net = CoFM_Large(num_class)
    else:
        raise NotImplementedError()

    return net
