import tensorflow as tf
from importlib import import_module
import os

from option import args

assert args.hwc is not None

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

with tf.Graph().as_default():
    #Build model
    model_module = import_module('model.' + args.model_type.lower())
    model_builder = model_module.make_model(args)
    model = model_builder.build()

    #Restore parameters
    root = tf.train.Checkpoint(model=model)
    checkpoint_dir = os.path.join(args.checkpoint_dir, model_builder.get_name())
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    #Save input, output tensor names to a config file
    with open(os.path.join(checkpoint_dir, 'config'), 'w') as f:
        f.write("{}\n".format(model.inputs[0].name))
        f.write("{}\n".format(model.outputs[0].name))

    #Save HDF5 file
    model.save(os.path.join(checkpoint_dir, 'final_{}_{}_{}.h5').format(args.hwc[0], args.hwc[1], args.hwc[2]))

    #Save fronzen graph (.pb) file
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        my_graph=tf.get_default_graph()
        frozen_graph = freeze_session(sess, output_names=[out.name for out in my_graph.get_operations()])
        tf.train.write_graph(frozen_graph, checkpoint_dir, 'final_{}_{}_{}.pb'.format(args.hwc[0], args.hwc[1], args.hwc[2]), as_text=False)
