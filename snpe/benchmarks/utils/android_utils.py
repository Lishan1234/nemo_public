#==============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

import os
import re
import subprocess
import time
from subprocess import check_output
from subprocess import CalledProcessError
from subprocess import STDOUT


UNKNOWN = 'unknown'
REGX_GET_PROP = re.compile('\[(.+)\]: \[(.+)\]')
REGEX_PROC_CPUINFO = re.compile('^Hardware.*Inc (.+)$')
getprop_list = ['ro.product.name',
                'ro.serialno',
                'ro.product.model',
                'ro.product.board',
                'ro.product.brand',
                'ro.product.device',
                'ro.product.manufacturer',
                'ro.product.cpu.abi',
                'ro.build.au_rev',
                'ro.build.description',
                'ro.build.version.sdk']
ADB_SHELL_CMD_SUCCESS = 'ADB_SHELL_CMD_SUCCESS'

class AdbShellCmdFailedException(Exception):
    def __str__(self):
        return('\nadb shell command Error: ' + repr(self) + '\n')

def get_device_list(hostname, logger):
    adb_cmd = 'adb devices'

    if hostname is not None:
        adb_cmd = 'adb -H ' + hostname + ' devices'
    device_list = []
    try:
        cmd_output = _execute_adbcmd_raw(adb_cmd, logger)
    except CalledProcessError as e:
        raise
    if cmd_output:
        regex_adb_devices = re.compile('^(.+)\s+device$')
        for line in cmd_output.split('\n'):
            m = regex_adb_devices.search(line)
            if m and m.group(1):
                device_list.append(m.group(1))
    return device_list


def get_device_info(device, logger, fatal=True):
    _info = {}
    getprop_cmd = "shell getprop | grep \"{0}\"".format('\|'.join(getprop_list))
    try:
        props = execute_adbcmd(device, getprop_cmd, logger)
    except CalledProcessError as e:
        if fatal != True:
            logger.warning('Non fatal get prop call failure, is the target os not Android?')
            return []
        raise
    if props:
        for line in props.split('\n'):
            m = REGX_GET_PROP.search(line)
            if m:
                _info[m.group(1)] = m.group(2)
    dev_info = []
    for prop_key in getprop_list:
        if not prop_key in _info:
            dev_info.append([prop_key, UNKNOWN])
        else:
            dev_info.append([prop_key, _info[prop_key]])
    return dev_info


def generate_adbcmd(device, cmd, shell=False):
    """
    Returns an adb command string that can be executed on the target
    """
    host = device.host_name
    dev_serial = device.comm_id
    if not shell:
        cmd_str = 'adb -H %s -s %s %s' % (host, dev_serial, cmd)
    else:
        cmd_str = 'adb -H %s -s %s shell \"%s && echo %s\"' % (host, dev_serial, cmd, ADB_SHELL_CMD_SUCCESS)
    return cmd_str

def _execute_adbcmd_raw(cmd_str, logger, shell=False, suppress_warning=False):
    cmd_handle = ''
    try:
        logger.debug('Executing {%s}' % cmd_str)
        p = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        cmd_out, cmd_err = p.communicate()
        cmd_out = cmd_out.decode()
        cmd_err = cmd_err.decode()
        returncode = p.returncode
        # check shell executed successfully on target
        if shell and (ADB_SHELL_CMD_SUCCESS not in cmd_out or returncode is not 0):
            # Between adb version 1.0.32 to 1.0.39, there is a change that now propagates
            # failed adb command properly (e.g. return code, also failure would show up
            # in stderr rather than stdout.  In order to work with both, we will need to
            # pass back both stdout and stderr to be processed by the caller, who may be
            # looking for particular substring (e.g. check for file existence).  Note this
            # is porcelain
            if not suppress_warning:
                logger.error('%s failed with stderr of: %s'%(cmd_str, cmd_out + cmd_err))
            raise AdbShellCmdFailedException(cmd_out + cmd_err)
        logger.debug('Command Output: \n"%s"' % cmd_out)
        return cmd_out
    except AdbShellCmdFailedException as e:
        if not suppress_warning:
            logger.warning('adb shell command failed to execute:\n\t%s' % repr(e))
        raise

def execute_adbcmd(device, cmd, logger, shell=False, suppress_warning=False):
    """
    Runs a BLOCKING adb command on target and raises exception
    when an error is encountered
    """
    cmd_str = generate_adbcmd(device, cmd, shell)
    return _execute_adbcmd_raw(cmd_str, logger, shell, suppress_warning)

def test_device_access(device, logger):
    """
    Tests access to android target connected to host device
    """
    access_cmd = 'get-state'
    try:
        cmd_handle = execute_adbcmd(device, access_cmd, logger)
        cmd_success = 'device' in cmd_handle.strip()
        if not cmd_success:
            logger.debug('Output from the get-state command %s' % cmd_handle)
            assert cmd_success, 'Could not run simple command on %s' % device
    except Exception as e:
        raise AdbShellCmdFailedException('Could not run simple command on device %s' % device.comm_id)

def device_prelim_settings(device, logger):
    """
    Turn off Sounds, Bluetooth, Wifi etc. services which can affect benchmark numbers
    """
    logger.info('Making prelim settings on device %s' % (device.comm_id))
    # Run ADB as root
    execute_adbcmd(device, 'root', logger)
    # Verify the flashed build image and warn if secondary image is not used
    uname_cmd = 'uname -a'
    try:
        uname_output = execute_adbcmd(device, uname_cmd, logger, True)
        is_perf_image = 'perf' in uname_output.strip()
        if not is_perf_image:
            logger.warning('Primary image flashed on %s. Its recommended to use Secondary image for running benchmarks' % device.comm_id)
    except Exception as e:
        logger.error(repr(e))
        raise AdbShellCmdFailedException('Could not run uname -a command on device %s' % device.comm_id)

    # Skip prelim settings for LE and AGL devices as they don't have APN mode and by default Bluetooth and WiFi are OFF for them
    is_LE_AGL = execute_adbcmd(device, 'shell getprop | grep "build.version.release"', logger)
    if is_LE_AGL == '':
        logger.info('LE or AGL dont require prelim settings, Device : %s' % (device.comm_id))
        return

    # Turn off Wifi
    wifi_disable_cmd = 'svc wifi disable'
    try:
        cmd_out = execute_adbcmd(device, wifi_disable_cmd, logger, True)
    except Exception as e:
        logger.error(repr(e))
        raise AdbShellCmdFailedException('Could not disbale wifi on device %s' % device.comm_id)

    ## Disable system sounds and vibration

    #Disable dialpad touch tone
    dtmf_tone_disable_cmd = 'settings put system dtmf_tone 0'
    #Disbale touch sound effects
    touch_sound_disable_cmd = 'settings put system sound_effects_enabled 0'
    #Disable screen lock sound
    screen_lock_sound_disable = 'settings put system lockscreen_sounds_enabled 0'
    #Disable vibrate on touch
    vibrate_on_touch_disable = 'settings put system haptic_feedback_enabled 0'
    #Fix screen brightness
    screen_brightness_fix_cmd = 'settings put system screen_brightness 30'
    #Disable screen auto-rotate
    screen_autorotate_disable_cmd = 'settings put system accelerometer_rotation 0'

    try:
        execute_adbcmd(device, dtmf_tone_disable_cmd, logger, True)
        execute_adbcmd(device, touch_sound_disable_cmd, logger, True)
        execute_adbcmd(device, screen_lock_sound_disable, logger, True)
        execute_adbcmd(device, vibrate_on_touch_disable, logger, True)
        execute_adbcmd(device, screen_brightness_fix_cmd, logger, True)
        execute_adbcmd(device, screen_autorotate_disable_cmd, logger, True)
    except Exception as e:
        raise AdbShellCmdFailedException('Could not disbale system sounds, vibration etc. on device %s' % device.comm_id)

    # Turn on Airplane mode
    apn_mode_cmd = 'settings put global airplane_mode_on'
    apn_mode_broadcast_cmd = 'am broadcast -a android.intent.action.AIRPLANE_MODE'
    try:
        #Turn off airplane mode and then turn it on, to make bluetooth off automatically
        execute_adbcmd(device, apn_mode_cmd + ' 0', logger, True)
        cmd_handle = execute_adbcmd(device, apn_mode_broadcast_cmd, logger, True)
        cmd_success = 'result=0' in cmd_handle.strip()
        if not cmd_success:
            logger.error('Output from the airplane mode broadcast command %s' % cmd_handle)
            assert cmd_success, 'Airplane mode broadcast command failed on %s' % device.comm_id
        time.sleep(3)
        execute_adbcmd(device, apn_mode_cmd + ' 1', logger, True)
        cmd_handle = execute_adbcmd(device, apn_mode_broadcast_cmd, logger, True)
        cmd_success = 'result=0' in cmd_handle.strip()
        if not cmd_success:
            logger.error('Output from the airplane mode broadcast command %s' % cmd_handle)
            assert cmd_success, 'Airplane mode broadcast command failed on %s' % device.comm_id
        #Verify airplane mode status
        airplane_mode_status_cmd = 'settings get global airplane_mode_on'
        apn_status = execute_adbcmd(device, airplane_mode_status_cmd, logger, True)
        apn_status = apn_status.splitlines()[0]
        if str(apn_status.strip()) != '1':
            logger.error('Output of airplane mode status is %s' % apn_status)
            assert apn_status == 1, 'Could not turn on airplane mode on %s' % device.comm_id
    except Exception as e:
        logger.error(repr(e))
        raise AdbShellCmdFailedException('Airplane mode toggle command failed to run on device %s' % device.comm_id)

    execute_adbcmd(device, 'unroot', logger)


def get_device_soc(device, logger):
    """
    Returns the SOC id of the device
    """
    group_id = 1
    cpuinfo_cmd = 'shell cat /proc/cpuinfo'
    ret = ''
    try:
        cmd_output = execute_adbcmd(device, cpuinfo_cmd, logger)
    except CalledProcessError as e:
        logger.error(repr(e))
        logger.error('cpuinfo command failed to execute on %s:\n\t%s' % (device.comm_id,cmd))
        raise
    if cmd_output:
        for line in cmd_output.split('\n'):
            m = REGEX_PROC_CPUINFO.search(line)
            if m and m.group(group_id):
                ret = m.group(group_id).rstrip().lstrip()
                break
    return ret


def fix_device_params(device, soc, logger):
    """
    Triggers the appropriate shell script based on SOC id to tune system params
    """
    logger.info('Tuning device params on device %s' % (device.comm_id))
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    scripts_dir = os.path.join(os.path.dirname(curr_dir), 'system_setting_scripts')
    assert os.path.exists(scripts_dir), 'System setting scripts directory path does not exist'
    script_name = soc + '.sh'
    host = device.host_name
    adb = 'adb -H %s' % (host)
    script_path = os.path.join(scripts_dir, script_name)
    if os.path.exists(script_path):
        try:
            logger.debug('Executing {%s}' % script_path)
            p = subprocess.Popen([script_path, adb, device.comm_id], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            p.wait()
            cmd_out, cmd_err = p.communicate()
            cmd_out = cmd_out.decode()
            cmd_err = cmd_err.decode()
            returncode = p.returncode
            if returncode != 0:
                logger.error('System setting script failed to run on device %s with Return code : %s' % (device.comm_id,str(returncode)))
                assert False, 'Command Output: \n"%s' % cmd_out
            logger.debug('Command Output: \n"%s"' % cmd_out)
        except AdbShellCmdFailedException as e:
            logger.error('System setting script failed to execute on %s:\n\t%s' % (device.comm_id,cmd))
            raise
    else:
        assert False, 'Could not find system setting script for Device id %s with SOC %s at location %s' % (device.comm_id, soc, scripts_dir)


def check_file_exists(device, file_path, logger, suppress_warning=False):
    """
    Returns 'True' if the file exists on the target
    """
    shell_cmd = "ls %s" % file_path
    try:
        execute_adbcmd(device, shell_cmd, logger, shell=True, suppress_warning=suppress_warning)
    except AdbShellCmdFailedException as e:
        if 'No such file or directory' in repr(e):
            return False
        else: # throw exception at caller: some other issue occurred
            raise
    else:
        # ls returning 0 means file/directory exists
        return True


def push_file(device, host_src_path, device_dest_dir, logger, silent=False):
    """
    A summary of the function

    Args:
        host_src_path: file/dir to be pushed
        device_dest_dir: destination folder on device
        device: DriodDevice object
        logger: logger object
        silent: To pipe the stdout to null

    Returns:
        True if the push is successful

    Raises:
        RuntimeError when the source_path does not exist
    """
    if not os.path.exists(host_src_path):
        logger.error('Path %s does not exist' % host_src_path)
        raise('%s is not a file or directory', RuntimeError)
    else:
        dir_exists = '[ -d %s ]' % (device_dest_dir)
        cmd_str = generate_adbcmd(device, dir_exists, True)
        p = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        cmd_out, cmd_err = p.communicate()
        if p.returncode == 0:    #if directory doesn't exist #mobinas
            mkdir_cmd = 'mkdir -p %s' % (device_dest_dir)
            cmd_handle = execute_adbcmd(device, mkdir_cmd, logger, True)
            logger.debug('Created directory %s on %s' % (device_dest_dir, device))
        src_name = os.path.basename(host_src_path)
        device_dest_path = os.path.join(device_dest_dir, src_name)
        logger.debug('Pushing %s to %s on %s' % (host_src_path, device_dest_path, device))
        push_cmd = 'push %s %s' % (host_src_path, device_dest_path)
        if silent:
            push_cmd += ' >/dev/null 2>&1'
        cmd_handle = execute_adbcmd(device, push_cmd, logger)
        logger.debug('Pushed %s to %s' % (host_src_path, device))
        return cmd_handle


def pull_file(device, device_src_path, host_dest_dir, logger, silent=False):
    """
    A summary of the function

    Args:
        device_src_path: file/dir to be pulled
        host_dest_dir: destination folder path
        device: DriodDevice object
        logger: logger object
        silent: To pipe the stdout to null

    Returns:
        True if the pull is successful
    """
    src_name = os.path.basename(device_src_path)
    host_dest_path = os.path.join(host_dest_dir, src_name)

    logger.debug('Pulling %s to %s from %s' % (device_src_path, host_dest_path, device))

    pull_cmd = 'pull %s %s' % (device_src_path, host_dest_path)

    if silent:
        pull_cmd += ' >/dev/null 2>&1'

    cmd_handle = execute_adbcmd(device, pull_cmd, logger)

    # catching errors
    no_error = cmd_handle.find('error') == -1
    file_found = cmd_handle.find('does not exist') == -1
    try:
        assert no_error, 'adb pull command threw an error on %s' % device
        assert file_found, 'Could not find the file %s on %s' % (device_src_path, device)
    except Exception as e:
        logger.error('Could not run pull command on %s' % device, exc_info=True)
        logger.error(repr(e))
        raise
    else:
        logger.info('Pulled %s to %s' % (device_src_path, host_dest_dir))
        return cmd_handle, host_dest_path
