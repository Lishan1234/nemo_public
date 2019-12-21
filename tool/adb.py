import os

def adb_rm(device_dir, device_id=None):
    if device_id:
        cmd = 'adb -s {} shell "rm -r {}"'.format(device_id, device_dir)
    else:
        cmd = 'adb shell "rm -r {}"'.format(device_dir)
    os.system(cmd)

def adb_mkdir(device_dir, device_id=None):
    if device_id:
        cmd = 'adb -s {} shell "mkdir -p {}"'.format(device_id, device_dir)
    else:
        cmd = 'adb shell "mkdir -p {}"'.format(device_dir)
    os.system(cmd)

def adb_push(device_filepath, host_filepath, device_id=None):
    if device_id:
        cmd = 'adb -s {} push {} {}'.format(device_id, host_filepath, device_filepath)
    else:
        cmd = 'adb push {} {}'.format(host_filepath, device_filepath)
    os.system(cmd)
    print(cmd)

def adb_pull(device_filepath, host_filepath, device_id=None):
    if device_id:
        cmd = 'adb -s {} pull {} {}'.format(device_id, device_filepath, host_filepath)
    else:
        cmd = 'adb pull {} {}'.format(device_filepath, host_filepath)
    os.system(cmd)
