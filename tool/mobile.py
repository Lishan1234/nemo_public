def playback_time(current, device_name):
    if device_name == 'xiaomi_mi9':
        capacity = 3300
        return 3300 / current
    elif device_name == 'xiaomi_redmi_note7':
        capacity = 4000
        return 4000 / current
    elif device_name == 'lg_gpad5':
        capacity = 8200
        return 8200 / current
    else:
        raise NotImplementedError

def id_to_name(id):
    if id == '7b7f59d1':
        return 'xiaomi_mi9'
    elif id == '10098e40':
        return 'xiaomi_redmi_note7'
    elif id == 'LMT605728961d9':
        return 'lg_gpad5'
    elif id == '637056d1':
        return 'xiaomi_note6_pro'
    else:
        raise NotImplementedError
