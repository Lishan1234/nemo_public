import os, time, sys, time
from importlib import import_module
import subprocess
import sys

import tensorflow as tf

from tool.common import freeze_session, optimize_for_inference

class SNPE():
    def __init__(self, model, checkpoint_dir, image_dir, log_dir, snpe_dir, hwc):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.image_dir = image_dir
        self.log_dir = log_dir
        self.snpe_dir = snpe_dir
        self.hwc = hwc

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

    #TODO
    def _snpe_dlc_viewer(self, dlc_name):
        setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(self.snpe_dir, self.tensorflow_dir)
        snpe_cmd = 'python {}/bin/x86_64-linux-clang/snpe-dlc-viewer\
                -i {} \
                -s {}.html'.format(self.snpe_dir, \
                                os.path.join(self.checkpoint_dir, dlc_name), \
                                os.path.join(self.checkpoint_dir, dlc_name))

        cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout)

    def _snpe_tensorflow_to_dlc(self, pb_name, input_name, output_name, dlc_name):
        setup_cmd = 'source {}/bin/envsetup.sh -t {}'.format(self.snpe_dir, self.tensorflow_dir)
        snpe_cmd = 'python {}/bin/x86_64-linux-clang/snpe-tensorflow-to-dlc \
                -i {} \
                --input_dim {} {} \
                --out_node {} \
                -o {} \
                --allow_unconsumed_nodes'.format(self.snpe_dir, \
                                os.path.join(self.checkpoint_dir, pb_name), \
                                input_name, \
                                '1,{},{},{}'.format(self.hwc[0], self.hwc[1], self.hwc[2]), \
                                output_name, \
                                os.path.join(self.checkpoint_dir, dlc_name))
        #TODO: allow_unsoncumed_nodes
        cmd = '{}; {}'.format(setup_cmd, snpe_cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
        proc_stdout = process.communicate()[0].strip()
        print(proc_stdout)

    #TODO: get filename from tf.train.Checkpoint and save as {name}.pb, {name}_opt.pb
    def convert(self):
        #restore
        ckpt = tf.train.Checkpoint(model=self.model)
        latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
        if not latest_ckpt:
            raise RuntimeError('checkpoint does not exist: {}'.format(self.checkpoint_dir))

        ckpt_name = os.path.basename(latest_ckpt).split('.')[0]
        pb_name = '{}.pb'.format(ckpt_name)
        opt_pb_name = '{}_opt.pb'.format(ckpt_name)
        dlc_name = '{}.dlc'.format(ckpt_name)

        #check dlc exists
        if os.path.exists(dlc_name):
            return dlc_name

        #save a frozen graph (.pb)
        status = ckpt.restore(latest_ckpt)
        sess = tf.keras.backend.get_session()
        status.initialize_or_restore(sess)
        graph = tf.get_default_graph()
        frozen_graph = freeze_session(sess, output_names=[out.op.name for out in self.model.outputs])
        tf.train.write_graph(frozen_graph, self.checkpoint_dir, pb_name, as_text=False)
        sess.close()

        #optimize a frozen graph
        input_name = self.model.inputs[0].name.split(':')[0]
        output_name = self.model.outputs[0].name.split(':')[0]
        optimize_for_inference(pb_name, opt_pb_name, input_name, output_name, self.checkpoint_dir)

        #convert to a dlc (.dlc)
        self._snpe_tensorflow_to_dlc(opt_pb_name, input_name, output_name, dlc_name)

        #visualize a dlc
        self._snpe_dlc_viewer(dlc_name)

        return dlc_name

    def convert_dataset(image_dir):
        #TODO: convert png images into SNPE applicable raw format (8bit 32bit)
        pass


    def setup(image_dir, ):
        #TODO: 1. copy images, models to a target device, 2.write a dataset list (txt file)
        #TODO: check whether data is already loaded on a device

        #data
        #host (images, model): model, checkpoint_dir, image_dir
        #host (image list): {image_dir}/snpe/target_raw_list.txt
        #device: {checkpoint_dir}/model, {image_dir}/model

        pass


    def evaluate(self):
        #TODO: 1. copy resulted images, 2. convert to png files, 3. calculate psnr
        #note: support all runtimes and quantization and power modes

        #result
        #host (images): {image_dir}/{model_name}.dlc/*.raw
        #host (logs): {log_dir}/{model_name}/snpe/benchmark_{soc name}_{runtime}_{etc}.json

        #result (quality)
        #host (logs): {log_dir}/{model_name}/snpe/qualiaty_{soc name}_{runtime}_{etc}.log
        pass

if __name__ == '__main__':
    pass
