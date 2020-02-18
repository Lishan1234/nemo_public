def count_mac_for_nas_s(model_name, height, width):
    if height == 240 and width == 426:
        if model_name == 'NAS_S_B8_F9_S4_deconv':
            return 3552 * 1000 * 1000
        elif model_name == 'NAS_S_B8_F21_S4_deconv':
            return 31555 * 1000 * 1000
        elif model_name == 'NAS_S_B8_F32_S4_deconv':
            return 101274 * 1000 * 1000
        elif model_name == 'NAS_S_B8_F48_S4_deconv':
            return 320966 * 1000 * 1000
        else:
            return None

    else:
        return None

def count_mac_for_nemo_s(model_name, height, width):
    if height == 240 and width == 426:
        if model_name == 'NEMO_S_B8_F9_S4_deconv':
            return 1913 * 1000 * 1000
        elif model_name == 'NEMO_S_B8_F21_S4_deconv':
            return 10337 * 1000 * 1000
        elif model_name == 'NEMO_S_B8_F32_S4_deconv':
            return 23958 * 1000 * 1000
        elif model_name == 'NEMO_S_B8_F48_S4_deconv':
            return 53840 * 1000 * 1000
        else:
            return None
    else:
        return None

def count_mac_for_dnn(model_name, height, width):
    if model_name.startswith('NAS_S'):
        return count_mac_for_nas_s(model_name, height, width)
    elif model_name.startswith('NEMO_S'):
        return count_mac_for_nemo_s(model_name, height, width)
    else:
        return None

def count_mac_for_cache(height, width, channel):
    return width * height * channel * 8
