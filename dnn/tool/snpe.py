import os, time, sys, time
from importlib import import_module
import subprocess
import sys
import glob
import logging
import scipy

import numpy as np
from PIL import Image
import tensorflow as tf

from tool.common import freeze_session, optimize_for_inference, check_attached_devices

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

SNPE_TARGET_ARCH        = 'arm-android-clang6.0'
SNPE_TARGET_STL         = 'libc++_shared.so'

class SNPE():
    raw_subdir = 'snpe_raw'
    dlc_subdir = 'snpe_dlc'
    log_subdir = 'snpe_log'

    device_rootdir = '/data/local/tmp/mobinas'
    device_lib_dir = os.path.join(device_rootdir, SNPE_TARGET_ARCH, 'lib')
    device_dsp_dir = os.path.join(device_rootdir, 'dsp', 'lib')
    device_bin_dir = os.path.join(device_rootdir, SNPE_TARGET_ARCH, 'bin')
    device_raw_dir = os.path.join(device_rootdir, raw_subdir)
    device_dlc_dir = os.path.join(device_rootdir, dlc_subdir)
    device_log_dir = os.path.join(device_rootdir, log_subdir)
    device_output_dir = os.path.join(device_rootdir, 'output')

    output_filename = 'raw_list.txt'
    library_filename = 'snpe.txt' #check for snpe library version
    dataset_filename = 'dataset.txt' #check for checkpoint, image versions

    def __init__(self, model, scale, hwc, image_dir, ref_image_dir, ckpt_dir, script_dir, snpe_dir):
        if not os.path.exists(image_dir):
            raise ValueError('image_dir does not exist: {}'.format(image_dir))
        if not os.path.exists(ckpt_dir):
            raise ValueError('ckpt_dir does not exist: {}'.format(ckpt_dir))
        if not os.path.exists(script_dir):
            raise ValueError('script_dir does not exist: {}'.format(script_dir))
        if not os.path.exists(snpe_dir):
            raise ValueError('snpe_dir does not exist: {}'.format(snpe_dir))

        self.model = model
        self.scale = scale
        self.hwc = hwc
        self.image_dir = image_dir
        self.ref_image_dir = ref_image_dir
        self.ckpt_dir = ckpt_dir
        self.script_dir = script_dir
        self.snpe_dir = snpe_dir

        self.dlc_dir = os.path.join(self.ckpt_dir, self.dlc_subdir)
        self.raw_dir = os.path.join(self.image_dir, self.raw_subdir)

        #check python version
        python_version = sys.version_info
        if not (python_version[0] == 3 and python_version[1] == 4):
            raise RuntimeError('Unsupported Python version: {}'.format(python_version))

        #check tensorflow version, location
        cmd = 'pip show tensorflow'
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = proc.stdout.readlines()
        self.tensorflow_dir = None
        for line in lines:
            line = line.decode().rstrip('\r\n')
            if 'Version' in line:
                tensorflow_ver = line.split(' ')[1]
                if not tensorflow_ver.startswith('1.13'):
                    raise RuntimeError('Tensorflow verion is wrong: {}'.format(tensorflow_ver))
            if 'Location' in line:
                tensorflow_dir = line.split(' ')[1]
                self.tensorflow_dir = os.path.join(tensorflow_dir, 'tensorflow')
        if tensorflow_dir is None:
            raise RuntimeError('Tensorflow is not installed')

    def _snpe_dlc_viewer(self, dlc_filepath, html_filepath):
        setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(self.snpe_dir, self.tensorflow_dir)
        snpe_cmd = 'python {}/bin/x86_64-linux-clang/snpe-dlc-viewer\
                -i {} \
                -s {}'.format(self.snpe_dir, \
                                dlc_filepath, \
                                html_filepath)

        cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout)

    def _snpe_tensorflow_to_dlc(self, pb_filepath, dlc_filepath, input_name, output_name):
        setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(self.snpe_dir, self.tensorflow_dir)
        snpe_cmd = 'python {}/bin/x86_64-linux-clang/snpe-tensorflow-to-dlc \
                -i {} \
                --input_dim {} {} \
                --out_node {} \
                -o {} \
                --allow_unconsumed_nodes'.format(self.snpe_dir, \
                                pb_filepath, \
                                input_name, \
                                '1,{},{},{}'.format(self.hwc[0], self.hwc[1], self.hwc[2]), \
                                output_name, \
                                dlc_filepath)
        cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout)

    def convert_model(self):
        #restore
        ckpt = tf.train.Checkpoint(model=self.model)
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        if not latest_ckpt:
            raise RuntimeError('checkpoint does not exist: {}'.format(self.ckpt_dir))

        ckpt_filename = os.path.basename(latest_ckpt).split('.')[0]
        pb_filename = '{}.pb'.format(ckpt_filename)
        opt_pb_filename = '{}_opt.pb'.format(ckpt_filename)
        dlc_filename = '{}.dlc'.format(ckpt_filename)
        qnt_dlc_filename = '{}_quantized.dlc'.format(ckpt_filename)

        dlc_dict = {'model_name': self.model.name , \
                'input_name': self.model.inputs[0].name, \
                'output_name': self.model.outputs[0].name, \
                'qnt_dlc_filename': qnt_dlc_filename, \
                'dlc_filename': dlc_filename}

        #check dlc exists
        if os.path.exists(self.dlc_dir):
            logging.info('dlc exists')
            return dlc_dict

        #save a frozen graph (.pb)
        status = ckpt.restore(latest_ckpt)
        sess = tf.keras.backend.get_session()
        status.initialize_or_restore(sess)
        graph = tf.get_default_graph()
        frozen_graph = freeze_session(sess, output_names=[out.op.name for out in self.model.outputs])
        tf.train.write_graph(frozen_graph, self.dlc_dir, pb_filename, as_text=False)

        #optimize a frozen graph
        input_name = self.model.inputs[0].name.split(':')[0]
        output_name = self.model.outputs[0].name.split(':')[0]
        pb_filepath = os.path.join(self.dlc_dir, pb_filename)
        opt_pb_filepath = os.path.join(self.dlc_dir, opt_pb_filename)
        optimize_for_inference(pb_filepath, opt_pb_filepath, input_name, output_name)

        #convert to a dlc (.dlc)
        dlc_filepath = os.path.join(self.dlc_dir, dlc_filename)
        self._snpe_tensorflow_to_dlc(pb_filepath, dlc_filepath, input_name, output_name)

        #convcert to a quantized dlc (.quantized.dlc)
        #TODO

        #visualize a dlc
        html_filepath = os.path.join(self.dlc_dir, '{}.html'.format(dlc_filename))
        self._snpe_dlc_viewer(dlc_filepath, html_filepath)

        return dlc_dict

    @staticmethod
    def _get_image_raw(image_filepath):
        image_filepath = os.path.abspath(image_filepath)
        image = Image.open(image_filepath)
        image_ndarray = np.array(image) # read it
        if len(image_ndarray.shape) != 3:
            raise RuntimeError('Image shape' + str(image_ndarray.shape))
        if (image_ndarray.shape[2] != 3):
            raise RuntimeError('Require image with rgb but channel is %d' % image_ndarray.shape[2])
        # reverse last dimension: rgb -> bgr
        return image_ndarray

    def convert_dataset(self, save_uint8=False):
        #1. check dataset exists
        print(self.raw_dir)
        if os.path.exists(self.raw_dir):
            logging.info('raw exists')
            return self.raw_dir
        os.makedirs(self.raw_dir)

        #2. convert png images to raw images
        image_filepaths = sorted(glob.glob('{}/*.png'.format(self.image_dir)))
        raw_filepaths = []
        for image_filepath in image_filepaths:
            raw = self._get_image_raw(image_filepath)
            raw = raw.astype(np.float32)

            if save_uint8:
                raw = raw.astype(np.uint8)
            else:
                raw = raw.astype(np.float32)

            image_filepath = os.path.abspath(image_filepath)
            filename, ext = os.path.splitext(image_filepath)
            raw_filepath = os.path.basename(filename)
            raw_filepath += '.raw'
            raw.tofile(os.path.join(self.raw_dir, raw_filepath))
            raw_filepaths.append(raw_filepath)

        #3. create a image list file
        device_image_filepaths = list(map(lambda x: os.path.join(self.device_raw_dir, os.path.basename(x)), raw_filepaths))
        output_filepath = os.path.join(self.raw_dir, self.output_filename)
        with open(output_filepath, 'w') as f:
            f.write('\n'.join(device_image_filepaths))

    @staticmethod
    def _adb_remove_dir(device_dir, device_id=None):
        if device_id:
            cmd = 'adb -s {} shell "rm -r {}"'.format(device_id, device_dir)
        else:
            cmd = 'adb shell "rm -r {}"'.format(device_dir)
        os.system(cmd)

    @staticmethod
    def _adb_make_dir(device_dir, device_id=None):
        if device_id:
            cmd = 'adb -s {} shell "mkdir -p {}"'.format(device_id, device_dir)
        else:
            cmd = 'adb shell "mkdir -p {}"'.format(device_dir)
        os.system(cmd)

    @staticmethod
    def _adb_push_file(device_filepath, host_filepath, device_id=None):
        if device_id:
            cmd = 'adb -s {} push {} {}'.format(device_id, host_filepath, device_filepath)
        else:
            cmd = 'adb push {} {}'.format(host_filepath, device_filepath)
        os.system(cmd)
        print(cmd)

    @staticmethod
    def _adb_pull_file(device_filepath, host_filepath, device_id=None):
        if device_id:
            cmd = 'adb -s {} pull {} {}'.format(device_id, device_filepath, host_filepath)
        else:
            cmd = 'adb pull {} {}'.format(device_filepath, host_filepath)
        os.system(cmd)

    #Note: assume arm64-v8a
    def setup_library(self, device_id=None):
        host_library_filepath = os.path.join(self.snpe_dir, self.library_filename) #temporally save at snpe_dir
        device_library_filepath = os.path.join(self.device_rootdir, self.library_filename)
        copy_library = True

        #1. check library exists on a target device
        self._adb_pull_file(device_library_filepath, host_library_filepath)
        if os.path.exists(host_library_filepath):
            with open(host_library_filepath, 'r') as f:
                line = f.readline()
                line = line.rstrip('\r\n')
                if line == self.snpe_dir:
                    copy_library = False
                    logging.info('library exists in a target_device')
            os.remove(host_library_filepath)

        #2. copy library
        if copy_library:
            self._adb_make_dir(self.device_rootdir, device_id)
            self._adb_remove_dir(self.device_lib_dir, device_id)
            self._adb_make_dir(self.device_lib_dir, device_id)
            self._adb_remove_dir(self.device_dsp_dir, device_id)
            self._adb_make_dir(self.device_dsp_dir, device_id)
            self._adb_remove_dir(self.device_bin_dir, device_id)
            self._adb_make_dir(self.device_bin_dir, device_id)

            self._adb_push_file(self.device_lib_dir, \
                                os.path.join(self.snpe_dir, 'lib', SNPE_TARGET_ARCH, SNPE_TARGET_STL))
            host_lib_dir = os.path.join(self.snpe_dir, 'lib', SNPE_TARGET_ARCH)
            host_filepaths = glob.glob('{}/*.so'.format(host_lib_dir))
            for host_filepath in host_filepaths:
                self._adb_push_file(self.device_lib_dir, host_filepath)
            host_dsp_dir = os.path.join(self.snpe_dir, 'lib', 'dsp')
            host_filepaths = glob.glob('{}/*.so'.format(host_dsp_dir))
            for host_filepath in host_filepaths:
                self._adb_push_file(self.device_dsp_dir, host_filepath)
            self._adb_push_file(self.device_bin_dir, \
                                os.path.join(self.snpe_dir, 'bin', SNPE_TARGET_ARCH, 'snpe-net-run'))

        #3. write a log file
        if copy_library:
            with open(host_library_filepath, 'w') as f:
                f.write(self.snpe_dir)
            self._adb_push_file(self.device_rootdir, host_library_filepath, device_id)
            os.remove(host_library_filepath)

    def setup_dataset(self, device_id=None):
        if not os.path.exists(self.dlc_dir):
            raise ValueError('dlc_dir does not exist: {}'.format(self.dlc_dir))
        if not os.path.exists(self.raw_dir):
            raise ValueError('raw_dir does not exist: {}'.format(self.raw_dir))

        copy_raw = True
        copy_dlc = True
        host_dataset_filepath = os.path.join(self.raw_dir, self.dataset_filename) #temporally save at raw_dir
        device_dataset_filepath = os.path.join(self.device_rootdir, self.dataset_filename)

        #1. check dataset exists on a target device
        self._adb_pull_file(device_dataset_filepath, host_dataset_filepath)
        if os.path.exists(host_dataset_filepath):
            with open(host_dataset_filepath, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.rstrip('\r\n')
                    print(line)
                    if line.split('\t')[0] == self.dlc_subdir and line.split('\t')[1] == self.dlc_dir:
                        logging.info('dlc exists in a target device')
                        copy_dlc = False
                    if line.split('\t')[0] == self.raw_subdir and line.split('\t')[1] == self.raw_dir:
                        logging.info('raw exists in a target device')
                        copy_raw = False
            os.remove(host_dataset_filepath)

        #2. remove exisiting device directory (.../checkpoint/, .../image)
        self._adb_make_dir(self.device_rootdir)

        #3. copy checkpoint and images
        if copy_raw:
            self._adb_remove_dir(self.device_raw_dir, device_id)
            self._adb_make_dir(self.device_raw_dir, device_id)
            self._adb_push_file(self.device_raw_dir, self.raw_dir, device_id)
        if copy_dlc:
            self._adb_remove_dir(self.device_dlc_dir, device_id)
            self._adb_make_dir(self.device_dlc_dir, device_id)
            self._adb_push_file(self.device_dlc_dir, self.dlc_dir, device_id)

        #4. write a log & push it into a target device
        if copy_raw or copy_dlc:
            with open(host_dataset_filepath, 'w') as f:
                f.write('{}\t{}\n'.format(self.dlc_subdir, self.dlc_dir))
                f.write('{}\t{}\n'.format(self.raw_subdir, self.raw_dir))
            self._adb_push_file(self.device_rootdir, host_dataset_filepath, device_id)
            os.remove(host_dataset_filepath)

    #TODO: let's move to common
    @staticmethod
    def _test_dataset(sr_image_dir, hr_image_dir):
        sr_image_files = sorted(glob.glob('{}/*.png'.format(sr_imgae_dir)))
        sr_ds = tf.data.Dataset.from_tensor_slices(sr_image_files)
        sr_ds = sr_ds.map(tf.io.read_file)
        sr_ds = sr_ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)

        hr_image_files = sorted(glob.glob('{}/*.png'.format(hr_imgae_dir)))
        hr_ds = tf.data.Dataset.from_tensor_slices(hr_image_files)
        hr_ds = hr_ds.map(tf.io.read_file)
        hr_ds = hr_ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)

        ds = tf.data.Dataset.zip((sr_ds, hr_ds))
        ds = ds.repeat(1)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def execute(self, runtime, dlc_dict, save_uint8=False):
        #TODO: test uint8, float images for a quantized model
        #TODO: cover CPU, AIP
        if runtime not in ['GPU_FP32', 'GPU_FP16']:
            raise ValueError('runtime is not supported: {}'.format(runtime))

        if runtime == 'GPU_FP16':
            runtime_opt = '--use_gpu --gpu_mode float16'
            dlc_filename = dlc_dict['dlc_filename']
        elif runtime == 'GPU_FP32':
            runtime_opt = '--use_gpu --gpu_mode default'
            dlc_filename = dlc_dict['dlc_filename']

        device_output_dir = os.path.join(self.device_output_dir, runtime)
        host_output_dir = os.path.join(self.image_dir, runtime)
        cmds = ['#!/system/bin/sh',
                'export SNPE_TARGET_ARCH={}'.format(SNPE_TARGET_ARCH),
                'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}/$SNPE_TARGET_ARCH/lib'.format(self.device_rootdir),
                'export PATH=$PATH:{}/$SNPE_TARGET_ARCH/bin'.format(self.device_rootdir),
                'rm -r {}'.format(device_output_dir),
                'mkdir -p {}'.format(device_output_dir),
                #'cd {}'.format(os.path.join(self.device_o, args.train_data, 'data')),
                #'snpe-net-run -h',
                'snpe-net-run --container {} {} --input_list {} --output_dir {}'.format(os.path.join(self.device_dlc_dir, dlc_filename), runtime_opt, os.path.join(self.device_raw_dir, self.output_filename), device_output_dir),
                'exit']

        cmd_script_filename = '{}.sh'.format(runtime)
        cmd_script_filepath = os.path.join(self.script_dir, cmd_script_filename)
        with open(cmd_script_filepath, 'w') as cmd_script:
            for ln in cmds:
                cmd_script.write(ln + '\n')
        #os.system('adb push {} {}'.format(cmd_script_filepath, self.device_rootdir)) #TODO: refactor
        #os.system('adb shell sh {}'.format(os.path.join(self.device_rootdir, cmd_script_filename))) #TODO: refactor
        #self._adb_pull_file(device_output_dir, host_output_dir)

        #convert .raw into .png
        output_name = self.model.outputs[0].name
        result_filepath = os.path.join(output_name.split('/')[0], '{}.raw'.format(output_name.split('/')[1]))
        result_dirs = glob.glob(os.path.join(self.image_dir, runtime, 'Result*'))
        for result_dir in result_dirs:
            idx = int(os.path.basename(result_dir).split('_')[1]) #Result_XX
            raw_filepath = os.path.join(result_dir, \
                                        output_name.split('/')[0], \
                                        '{}.raw'.format(output_name.split('/')[1]))
            if save_uint8:
                arr = np.fromfile(raw_filepath, dtype=np.uint8)
                arr = np.reshape(arr, (self.hwc[0] * self.scale, self.hwc[1] * self.scale, self.hwc[2]))
                arr = np.clip(arr, 0, 255)
            else:
                arr = np.fromfile(raw_filepath, dtype=np.float32)
                arr = np.reshape(arr, (self.hwc[0] * self.scale, self.hwc[1] * self.scale, self.hwc[2]))
                arr = np.clip(arr, 0.0, 255.0)
                arr = np.round(arr)
            scipy.misc.imsave(os.path.join(self.image_dir, runtime, '{:04d}.png'.format(idx)), arr)
            os.rmdir(result_dir)

        #TODO: measure psnr
        """
        sr_image_dir = host_output_dir
        hr_image_dir = self.ref_image_dir

        lr_image_filepath

        image_filepaths = sorted(glob.glob('{}/*.png'.format(self.image_dir)))
        """

    #TODO
    def benchmark(self):
        pass

if __name__ == '__main__':
    pass
