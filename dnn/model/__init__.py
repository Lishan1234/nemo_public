from .nemo_s import NEMO_S

def build(model_type, num_blocks, num_filters, scale, upsample_type):
    if model_type == 'nemo_s':
        model = NEMO_S(num_blocks, num_filters, scale, upsample_type).build()
    else:
        raise NotImplementedError('Unsupported model: {}'.format(model_type))
    return model
