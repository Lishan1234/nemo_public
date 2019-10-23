import sys
import tensorflow as tf
import random
import shlex
import json
import subprocess
import os
import platform

OPT_4_INFERENCE_SCRIPT              = 'optimize_for_inference.py'

def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def load_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def crop_image(hr_image, lr_image, lr_bicubic_image, scale, patch_size):
    height, width, channel = lr_image.get_shape().as_list()
    #print(height, width)
    rand_height = random.randint(0, height - patch_size - 1)
    rand_width = random.randint(0, width - patch_size - 1)
    hr_image_cropped = tf.image.crop_to_bounding_box(hr_image,
                                                    rand_height * scale,
                                                    rand_width * scale,
                                                   patch_size * scale,
                                                   patch_size * scale)
    lr_image_cropped = tf.image.crop_to_bounding_box(lr_image,
                                                    rand_height,
                                                    rand_width,
                                                   patch_size,
                                                   patch_size)
    lr_bicubic_image_cropped = tf.image.crop_to_bounding_box(lr_bicubic_image,
                                                    rand_height * scale,
                                                    rand_width * scale,
                                                   patch_size * scale,
                                                   patch_size * scale)
    return hr_image_cropped, lr_image_cropped, lr_bicubic_image_cropped

def crop_augment_image(hr_image, lr_image, scale, patch_size):
    #Crop
    height, width, channel = lr_image.get_shape().as_list()
    rand_height = random.randint(0, height - patch_size - 1)
    rand_width = random.randint(0, width - patch_size - 1)
    hr_image_cropped = tf.image.crop_to_bounding_box(hr_image,
                                                    rand_height * scale,
                                                    rand_width * scale,
                                                   patch_size * scale,
                                                   patch_size * scale)
    lr_image_cropped = tf.image.crop_to_bounding_box(lr_image,
                                                    rand_height,
                                                    rand_width,
                                                   patch_size,
                                                   patch_size)

    #Augement
    if random.randint(0,1):
        hr_image_cropped = tf.image.flip_left_right(hr_image_cropped)
        lr_image_cropped = tf.image.flip_left_right(lr_image_cropped)
    if random.randint(0,1):
        hr_image_cropped = tf.image.flip_up_down(hr_image_cropped)
        lr_image_cropped = tf.image.flip_up_down(lr_image_cropped)
    for _ in range(random.randint(0,3)):
        hr_image_cropped = tf.image.rot90(hr_image_cropped)
        lr_image_cropped = tf.image.rot90(lr_image_cropped)

    return hr_image_cropped, lr_image_cropped

def findFPS(video_path):
    cmd = "/usr/bin/ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(video_path)
    ffprobeOutput = subprocess.check_output(args).decode("utf-8")
    ffprobeOutput = json.loads(ffprobeOutput)

    _fps = ffprobeOutput['streams'][0]['r_frame_rate'].split('/')
    fps = float(_fps[0]) / float(_fps[1])

    return fps

def findWidthHeight(resolution):
    if resolution == 270:
        return 1080, 270
    else:
        raise NotImplementedError

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ''
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

def find_optimize_for_inference():
    cmd = 'pip show tensorflow'
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = proc.stdout.readlines()

    tensorflow_dir = None
    for line in lines:
        line = line.decode().rstrip('\r\n')
        if 'Location' in line:
            tensorflow_dir = line.split(' ')[1]

            for root, dirs, files in os.walk(os.path.join(tensorflow_dir, 'tensorflow')):
                if OPT_4_INFERENCE_SCRIPT in files:
                    return os.path.join(root, OPT_4_INFERENCE_SCRIPT)

            break

    return None

def optimize_for_inference(pb_filename, input_name, output_name, checkpoint_dir):
    # Try to optimize the inception v3 PB for inference
    opt_4_inference_file = find_optimize_for_inference()

    if not opt_4_inference_file:
        print("\nWARNING: cannot find " + OPT_4_INFERENCE_SCRIPT + " script. Skipping inference optimization.\n")
    else:
        print('INFO: Optimizing for inference Inception v3 using ' + opt_4_inference_file)
        print('      Please wait. It could take a while...')
        cmd = ['python', opt_4_inference_file,
               '--input', os.path.join(checkpoint_dir, pb_filename),
               '--output', os.path.join(checkpoint_dir, pb_filename + '_opt'),
               '--input_names', input_name,
               '--output_names', output_name]
        subprocess.call(cmd)
        pb_filename = pb_filename + '_opt'

    return pb_filename

#Caution: needs system python to install tensorflow
def convert_to_dlc(pb_filename, input_name, output_name, dlc_filename, checkpoint_dir, h, w, c):
    """
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = '{}:../third_party/snpe/lib/python'.format(os.environ['PYTHONPATH'])
    else:
        os.environ['PYTHONPATH'] = '../third_party/snpe/lib/python'
    """

    print('INFO: Converting ' + pb_filename +' to SNPE DLC format')
    cmd = ['/usr/bin/python2',
           '../third_party/snpe/bin/x86_64-linux-clang/snpe-tensorflow-to-dlc',
           '--graph', os.path.join(checkpoint_dir, pb_filename),
           '--input_dim', input_name, '1,{},{},{}'.format(h, w, c),
           '--out_node', output_name,
           '--dlc', os.path.join(checkpoint_dir, dlc_filename),
           '--allow_unconsumed_nodes']
    subprocess.call(cmd)
