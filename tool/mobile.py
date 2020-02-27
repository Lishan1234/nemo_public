def playback_time(current, device_name):
    if device_name == 'mi9':
        capacity = 3300
        return 3300 / current
    elif device_name == 'redmi_note7':
        capacity = 4000
        return 4000 / current
    elif device_name == 'gpad':
        capacity = 8200
        return 8200 / current
    else:
        raise NotImplementedError
