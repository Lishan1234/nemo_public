import os, time, sys, time
from importlib import import_module
import subprocess
import sys
import glob

import numpy as np
from PIL import Image
import tensorflow as tf

from tool.common import freeze_session, optimize_for_inference, check_attached_devices

class SNPE():
    device_rootdir = '/data/local/tmp/mobinas'
    device_imagedir = os.path.join(device_rootdir, 'image')
    device_checkpointdir =os.path.join(device_rootdir, 'checkpoint')
    device_logdir = os.path.join(device_rootdir, 'log')
    output_filename = 'raw_list.txt'

    def __init__(self, snpe_dir):
        if not os.path.exists(snpe_dir):
            raise ValueError('snpe_dir does not exist: {}'.format(snpe_dir))

        self.snpe_dir = snpe_dir
        #TODO: extract snpe verison
        #TODO: setup snpe prerequisite for each SNPE version


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

    def _snpe_dlc_viewer(self, checkpoint_dir, dlc_filename):
        setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(self.snpe_dir, self.tensorflow_dir)
        snpe_cmd = 'python {}/bin/x86_64-linux-clang/snpe-dlc-viewer\
                -i {} \
                -s {}.html'.format(self.snpe_dir, \
                                os.path.join(checkpoint_dir, dlc_filename), \
                                os.path.join(checkpoint_dir, dlc_filename))

        cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout)

    def _snpe_tensorflow_to_dlc(self, checkpoint_dir, hwc, pb_filename, input_name, output_name, dlc_filename):
        setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(self.snpe_dir, self.tensorflow_dir)
        snpe_cmd = 'python {}/bin/x86_64-linux-clang/snpe-tensorflow-to-dlc \
                -i {} \
                --input_dim {} {} \
                --out_node {} \
                -o {} \
                --allow_unconsumed_nodes'.format(self.snpe_dir, \
                                os.path.join(checkpoint_dir, pb_filename), \
                                input_name, \
                                '1,{},{},{}'.format(hwc[0], hwc[1], hwc[2]), \
                                output_name, \
                                os.path.join(checkpoint_dir, dlc_filename))
        cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout)

    def convert_model(self, model, checkpoint_dir, hwc):
        if not os.path.exists(checkpoint_dir):
            raise ValueError('checkpoint_dir does not exist: {}'.format(checkpoint_dir))

        #restore
        ckpt = tf.train.Checkpoint(model=model)
        latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if not latest_ckpt:
            raise RuntimeError('checkpoint does not exist: {}'.format(checkpoint_dir))

        ckpt_filename = os.path.basename(latest_ckpt).split('.')[0]
        pb_filename = '{}.pb'.format(ckpt_filename)
        opt_pb_filename = '{}_opt.pb'.format(ckpt_filename)
        dlc_filename = '{}.dlc'.format(ckpt_filename)

        #check dlc exists
        if os.path.exists(dlc_filename):
            return dlc_filename

        #save a frozen graph (.pb)
        status = ckpt.restore(latest_ckpt)
        sess = tf.keras.backend.get_session()
        status.initialize_or_restore(sess)
        graph = tf.get_default_graph()
        frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(frozen_graph, checkpoint_dir, pb_filename, as_text=False)
        sess.close()

        #optimize a frozen graph
        input_name = model.inputs[0].name.split(':')[0]
        output_name = model.outputs[0].name.split(':')[0]
        optimize_for_inference(pb_filename, opt_pb_filename, input_name, output_name, checkpoint_dir)

        #convert to a dlc (.dlc)
        self._snpe_tensorflow_to_dlc(checkpoint_dir, hwc, opt_pb_filename, input_name, output_name, dlc_filename)

        #visualize a dlc
        self._snpe_dlc_viewer(checkpoint_dir, dlc_filename)

        return dlc_filename

    #TODO: convert png images into SNPE applicable raw format (8bit 32bit)
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

    def convert_dataset(self, image_dir, save_uint8):
        if not os.path.exists(image_dir):
            raise ValueError('image_dir does not exist: {}'.format(image_dir))

        image_filepaths = sorted(glob.glob('{}/*.png'.format(image_dir)))
        for image_filepath in image_filepaths:
            image_raw = self._get_image_raw(image_filepath)
            snpe_raw = image_raw
            snpe_raw = snpe_raw.astype(np.float32)

            if save_uint8:
                snpe_raw = snpe_raw.astype(np.uint8)
            else:
                snpe_raw = snpe_raw.astype(np.float32)

            image_filepath = os.path.abspath(image_filepath)
            filename, ext = os.path.splitext(image_filepath)
            snpe_raw_filename = filename
            snpe_raw_filename += '.raw'
            snpe_raw.tofile(snpe_raw_filename)

    #TODO
    def setup_library(self):
        pass

    def setup_dataset(self, image_dir, checkpoint_dir, device_id=None):
        if not os.path.exists(checkpoint_dir):
            raise ValueError('checkpoint_dir does not exist: {}'.format(checkpoint_dir))
        if not os.path.exists(image_dir):
            raise ValueError('image_dir does not exist: {}'.format(image_dir))
        if not check_attached_devices(device_id):
            raise RuntimeError('device is not attached: {}'.format(device_id))

        #1. create a image list file
        image_host_filepaths = sorted(glob.glob('{}/*.raw'.format(image_dir)))
        image_device_filepaths = list(map(lambda x: os.path.join(self.device_imagedir, os.path.basename(x)), image_host_filepaths))
        output_filepath = os.path.join(image_dir, self.output_filename)
        with open(output_filepath, 'w') as f:
            f.write('\n'.join(image_device_filepaths))

        #2. remove exisiting device directory (.../checkpoint/, .../image)
        adb_cmd = 'adb shell rm -r "{}"a'.format(self.device_rootdir)
        os.system(adb_cmd)
        adb_cmd = 'adb shell "mkdir -p {}"'.format(self.device_imagedir)
        os.system(adb_cmd)
        adb_cmd = 'adb shell "mkdir -p {}"'.format(self.device_logdir)
        os.system(adb_cmd)
        adb_cmd = 'adb shell "mkdir -p {}"'.format(self.device_checkpointdir)
        os.system(adb_cmd)

        #3. copy checkpoint and images
        adb_cmd = ''

        #data
        #host (images, model): model, checkpoint_dir, image_dir
        #host (image list): {image_dir}/snpe/target_raw_list.txt
        #device: {checkpoint_dir}/model, {image_dir}/model

    def evaluate(self):
        #TODO: 1. copy resulted images, 2. convert to png files, 3. calculate psnr
        #TODO: check quality (normalized_config: None)
        #note: support all runtimes and quantization and power modes

        #result
        #host (images): {image_dir}/{model_name}.dlc/*.raw
        #host (logs): {log_dir}/{model_name}/snpe/benchmark_{soc name}_{runtime}_{etc}.json

        #result (quality)
        #host (logs): {log_dir}/{model_name}/snpe/qualiaty_{soc name}_{runtime}_{etc}.log

        #quality

        #latency
        pass

if __name__ == '__main__':
    pass
