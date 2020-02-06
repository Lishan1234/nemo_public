# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from snpe.converters.common.converter_ir import op_adapter
from snpe.converters.common.converter_ir.axis_tracker import AxisTracker


class OpNode(object):
    def __init__(self, op, input_names, output_names):
        self.op = op
        self.input_names = input_names
        self.output_names = output_names


class Buffer(object):
    def __init__(self, name, shape, producer):
        self.name = name
        self.producer = producer
        self.consumers = set()
        self.shape = shape
        self.axis_format = AxisTracker.AxisFormat.NOT_YET_DEFINED

    def rank(self):
        return len(self.shape)

    def get_axis_order(self):
        """Translate AxisFormat enum to modeltools axis order list"""
        if self.axis_format == 'NSC':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH]
        if self.axis_format == 'NCS':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL]
        elif self.axis_format == 'FEATURE':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.FEATURE]
        elif self.axis_format == 'BTF':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.TIME,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif self.axis_format == 'NONTRIVIAL':
            return [AxisTracker.AxisAnnotations.NONTRIVIAL]
        else:
            raise ValueError("Encountered unexpected axis format for get_axis_order: %s" % self.axis_format)


class IROpGraph(object):
    def __init__(self, naming_policy, shape_inference_policy):
        self.naming_policy = naming_policy
        self.shape_inference_policy = shape_inference_policy

        self.nodes_by_name = {}
        self.nodes_in_order = []
        self.buffers = {}

    def __iter__(self):
        return iter(self.nodes_in_order)

    def __insert_node(self, node, output_shapes, idx=-1):
        """Insert a node into the graph's internal data structures.

        node: Node to be inserted
        output_shapes: shapes of the node's output buffers, which must be created.
        idx: index in nodes_in_order at which to insert. By default, appends to
             the list."""
        for name, shape in zip(node.output_names, output_shapes):
            self.buffers[name] = Buffer(name, shape, node)

        for name in node.input_names:
            self.buffers[name].consumers.add(node)

        self.nodes_by_name[node.op.name] = node
        if idx == -1:
            self.nodes_in_order.append(node)
        else:
            self.nodes_in_order.insert(idx, node)

    def add(self, op, input_names, output_names):
        op.name = self.naming_policy.get_op_name(op)

        if not isinstance(input_names, list):
            input_names = [input_names]
        input_names = self.naming_policy.get_input_names(op, input_names)

        input_shapes = []
        for name in input_names:
            if name not in self.buffers:
                raise KeyError("Graph has no buffer %s, referred to as input for %s" % (name, op.name))
            input_shapes.append(self.buffers[name].shape)

        if not isinstance(output_names, list):
            output_names = [output_names]
        output_names = self.naming_policy.get_output_names(op, output_names)

        node = OpNode(op, input_names, output_names)

        output_shapes = self.shape_inference_policy.infer_shape(op, input_shapes)
        if len(output_shapes) != len(output_names):
            raise ValueError("Op %s: produced %d output shapes, but have %d outputs" % (op.name, len(output_shapes),
                                                                                        len(output_names)))

        # at this point everything should be error free, so it's fine to actually
        # touch the data structures
        self.__insert_node(node, output_shapes)

    def replace(self, old_op, new_op):
        old_node = self.nodes_by_name[old_op.name]
        input_buffers = self.get_input_buffers(old_node)
        output_buffers = self.get_output_buffers(old_node)
        input_names = [buf.name for buf in input_buffers]
        output_names = [buf.name for buf in output_buffers]

        # Create OpNode for the new op
        new_op.name = self.naming_policy.get_op_name(new_op)
        new_node = OpNode(new_op, input_names, output_names)

        # Replace the op in buffers
        input_shapes = []
        for buf in input_buffers:
            buf.consumers.remove(old_node)
            buf.consumers.add(new_node)
            input_shapes.append(buf.shape)

        output_shapes = self.shape_inference_policy.infer_shape(new_op, input_shapes)
        for i, buf in enumerate(output_buffers):
            buf.producer = new_node
            buf.shape = output_shapes[i]

        # Replace the op in op-lists
        idx = self.nodes_in_order.index(old_node)
        self.nodes_by_name[new_op.name] = new_node
        if idx == -1:
            self.nodes_in_order.append(new_node)
        else:
            self.nodes_in_order.insert(idx, new_node)

        del self.nodes_by_name[old_node.op.name]
        self.nodes_in_order.remove(old_node)

    def add_input(self, name, shape, encoding, input_type):
        op = op_adapter.InputOp(name, shape,
                                image_encoding_in=encoding,
                                image_encoding_out=encoding,
                                image_type=input_type)
        output_names = self.naming_policy.get_output_names(op, [name])

        node = OpNode(op, [], output_names)
        self.__insert_node(node, [shape])

    def inject(self, op, input_name, output_name, consumer_names=None):
        op.name = self.naming_policy.get_op_name(op)
        if input_name not in self.buffers:
            raise KeyError("Cannot inject op %s onto nonexistent buffer %s" % (op.name, input_name))

        input_buffer = self.buffers[input_name]
        if consumer_names is None:
            old_consumers = list(input_buffer.consumers)
            input_buffer.consumers.clear()
        else:
            old_consumers = []
            for name in consumer_names:
                if name not in self.nodes_by_name:
                    raise KeyError("Cannot inject op %s with nonexistent consumer %s" % (op.name, name))
                consumer = self.nodes_by_name[name]
                if consumer not in input_buffer.consumers:
                    raise KeyError("Cannot inject op %s, specified consumer %s does not actually consume input"
                                   " buffer %s" % (op.name, name, input_name))

                old_consumers.append(consumer)
                input_buffer.consumers.remove(consumer)

        output_name = self.naming_policy.get_output_names(op, [output_name])[0]
        producer_idx = self.nodes_in_order.index(input_buffer.producer)
        output_shapes = self.shape_inference_policy.infer_shape(op, [input_buffer.shape])
        node = OpNode(op, [input_name], [output_name])
        self.__insert_node(node, output_shapes, producer_idx+1)

        output_buffer = self.buffers[output_name]
        for consumer in old_consumers:
            output_buffer.consumers.add(consumer)
            for i, name in enumerate(consumer.input_names):
                if name == input_name:
                    consumer.input_names[i] = output_name

    def prune(self, node):
        """Remove a node and its output buffers from the graph completely.
        Will raise an exception if the node has any successors."""

        output_buffers = self.get_output_buffers(node)
        consumers = []
        for buf in output_buffers:
            consumers.extend(buf.consumers)
        consumers = [c.op.name for c in consumers]
        if len(consumers) > 0:
            raise RuntimeError("Cannot prune node %s, which has the following successors: %s"
                               % (node.op.name, consumers))

        for buf in output_buffers:
            del self.buffers[buf.name]
        for buf in self.get_input_buffers(node):
            buf.consumers.remove(node)
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)

    def squash(self, node, input_name):
        # remove the input buffer, causing that buffer's
        # producer to producer the output buffer instead.
        if input_name not in self.buffers:
            raise KeyError("Cannot squash node %s onto non-existent input buffer %s" % (node.op.name, input_name))
        input_buffer = self.buffers[input_name]
        output_buffer = self.buffers[node.output_names[0]]

        if len(input_buffer.consumers) > 1:
            raise ValueError("Cannot squash node %s onto input buffer %s, which has more than one consumer"
                             % (node.op.name, input_name))
        if node not in input_buffer.consumers:
            raise ValueError("Cannot squash node %s onto input buffer %s that it doesn't consumer"
                             % (node.op.name, input_name))

        prev = input_buffer.producer
        output_idx = prev.output_names.index(input_name)
        prev.output_names[output_idx] = output_buffer.name
        output_buffer.producer = prev

        del self.buffers[input_name]
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)

    def get_input_buffers(self, node):
        return [self.buffers[name] for name in node.input_names]

    def get_output_buffers(self, node):
        return [self.buffers[name] for name in node.output_names]

    def get_buffer(self, buffer_name):
        return self.buffers[buffer_name]

    def list_nodes(self):
        return self.nodes_in_order[:]

    def list_buffers(self):
        return list(self.buffers.values())
