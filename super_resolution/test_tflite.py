import tensorflow as tf
from importlib import import_module
import os
import scipy.misc

from option import args

assert args.hwc is not None

#with tf.Graph().as_default():
#Build model
model_module = import_module('model.' + args.model_type.lower())
model_builder = model_module.make_model(args)
model = model_builder.build()

#Restore parameters
checkpoint_dir = os.path.join(args.checkpoint_dir, args.train_data, model_builder.get_name())
os.makedirs(checkpoint_dir, exist_ok=True)
root = tf.train.Checkpoint(model=model)
print(checkpoint_dir)
assert tf.train.latest_checkpoint(checkpoint_dir) is not None
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

#Save input, output tensor names to a config file
with open(os.path.join(checkpoint_dir, 'config'), 'w') as f:
    f.write("{}\n".format(model.inputs[0].name))
    f.write("{}\n".format(model.outputs[0].name))

#Save HDF5 file
keras_file=os.path.join(checkpoint_dir, 'final_{}_{}_{}.h5').format(args.hwc[0], args.hwc[1], args.hwc[2])
model.save(keras_file)

#Convert to tflite model
lite_model=os.path.join(checkpoint_dir, 'final_{}_{}_{}_quantized.tflite').format(args.hwc[0], args.hwc[1], args.hwc[2])
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_file)
converter.inference_type = tf.uint8
converter.quantized_input_stats = {"input_1": (127.5, 128.0)}
#converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open(lite_model, "wb").write(tflite_model)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()

# Read input/output images
hr_image_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}p/original'.format(args.hr))
lr_image_path = os.path.join(args.data_dir, args.train_data, args.data_type, '{}p/original'.format(args.hr//args.scale))
hr_image_filenames = sorted(glob.glob('{}/*.png'.format(hr_image_path)))
lr_image_filenames = sorted(glob.glob('{}/*.png'.format(lr_image_path)))

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_data = scipy.misc.imread(lr_image_filenames[0])
print(input_data)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
