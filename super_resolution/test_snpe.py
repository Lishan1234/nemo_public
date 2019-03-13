import tensorflow as tf
from importlib import import_module
import os
from option import args
import sys
#tf.enable_eager_execution()

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
        freeze_var_names = list(set(v.op.name for v in tf.trainable_variables()).difference(keep_var_names or []))
        #print(freeze_var_names)
        output_names = output_names or []
        #for v in tf.trainable_variables():
        #    print(v.op.name)
        output_names += [v.op.name for v in tf.trainable_variables()]
        print(output_names)
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

assert args.hwc is not None

tf.keras.backend.set_learning_phase(0)

#Build model
model_module = import_module('model.' + args.model_type.lower())
model_builder = model_module.make_model(args)
model = model_builder.build()

#Checkpoint
#root = tf.train.Checkpoint(model=model)
#checkpoint_prefix = os.path.join('/home/hyunho/MobiNAS/super_resolution', 'ckpt')
#checkpoint_prefix = 'checkpoint/ckpt'
#root.save(checkpoint_prefix)
#os.makedirs('checkpoint', exist_ok=True)
#root = tf.train.Checkpoint(model=model)
#root.restore(tf.train.latest_checkpoint('checkpoint'))

#Save fronzen graph (.pb) file
#Freeze a graph
sess = tf.keras.backend.get_session()
init = tf.global_variables_initializer()
sess.run(init)
my_graph=tf.get_default_graph()
#output_names=[out.name for out in my_graph.get_operations()]
#print(output_names)
frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, './', 'tmp.pb', as_text=False)

"""
#Build model
model_module = import_module('model.' + args.model_type.lower())
model_builder = model_module.make_model(args)
model = model_builder.build()

#Restore parameters
checkpoint_dir = os.path.join(args.checkpoint_dir, model_builder.get_name())
os.makedirs(checkpoint_dir, exist_ok=True)
if args.use_random_weights is not True:
    root = tf.train.Checkpoint(model=model)
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))

#Save fronzen graph (.pb) file
with tf.Session() as sess:
    sess.run(init)
    my_graph=tf.get_default_graph()
    frozen_graph = freeze_session(sess, output_names=[out.name for out in my_graph.get_operations()])
    tf.train.write_graph(frozen_graph, checkpoint_dir, 'final_{}_{}_{}.pb'.format(args.hwc[0], args.hwc[1], args.hwc[2]), as_text=False)
"""
