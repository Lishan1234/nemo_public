from .nemo_s import NEMO_S

def build(model_type, num_blocks, num_filters, scale, upsample_type, apply_clip=None, output_shape=None):
    if model_type == 'nemo_s':
        model = NEMO_S(num_blocks, num_filters, scale, upsample_type).build(output_shape=output_shape, apply_clip=apply_clip)
    else:
        raise NotImplementedError('Unsupported model: {}'.format(model_type))
    return model
