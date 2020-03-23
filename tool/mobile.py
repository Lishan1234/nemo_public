def playback_time(current, device_name):
    if device_name == 'mi9':
        capacity = 3300
        return 3300 / current
    elif device_name == 'redmi':
        capacity = 4000
        return 4000 / current
    elif device_name == 'gpad':
        capacity = 8200
        return 8200 / current
    else:
        raise NotImplementedError

def id_to_name(id):
    if id == '7b7f59d1':
        return 'mi9'
    elif id == '10098e40':
        return 'redmi'
    elif id == 'LMT605728961d9':
        return 'gpad'
    else:
        raise NotImplementedError
