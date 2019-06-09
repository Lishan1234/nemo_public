#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2018 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import numpy as np

from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphMatcher,
    GraphSequence,
    NonConsumableConverterSequenceNode
)

class InstanceNormLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, shape):
            super(InstanceNormLayerResolver.Descriptor, self).__init__('InstanceNorm', name, operations)
            self.shape = shape
            # SNPE runtime algo is y = x * WEIGHT / rms + BIAS
            # While L2 Normalization is y = x / rms
            # That requires WEIGHT = 1.0 and BIAS = 0.0 to mimic L2 Norm in SNPE
            self.weights = np.ones(shape)
            self.biases = np.zeros(shape)

    def __init__(self):
        # Graph topology of tf.math.l2_normalize
        self.sequence = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('a', ['Square']),
            ConverterSequenceNode('weights', ['Const', 'Identity']),
            ConverterSequenceNode('b', ['Sum']),
            ConverterSequenceNode('epsilon', ['Const', 'Identity']),
            ConverterSequenceNode('c', ['Maximum']),
            ConverterSequenceNode('d', ['Rsqrt']),
            ConverterSequenceNode('e', ['Mul'])
        ])
        self.sequence.set_inputs('a', ['input'])
        self.sequence.set_inputs('b', ['a', 'weights'])
        self.sequence.set_inputs('c', ['b', 'epsilon'])
        self.sequence.set_inputs('d', ['c'])
        self.sequence.set_inputs('e', ['d', 'input'])
        self.sequence.set_outputs(['e'])

    # For now, elementwise resolver cannot work with epsilon node.
    # Will meet error "ElementWise resolver must implement broadcast method.".
    def is_final_resolution(self):
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        potential_descriptors = []
        for match in matches:
            bn_op = match['a']
            input_op = match['input']

            shape = graph_helper.get_op_output_shape(input_op)

            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(InstanceNormLayerResolver.Descriptor(str(bn_op.name),
                                                           consumed_nodes,
                                                           shape=shape))
        return potential_descriptors

class InstanceNormLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: InstanceNormLayerBuilder.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        # Set `compute_statistics=True` to activate instance norm
        return converter_context.model.add_batchnorm_layer(descriptor.layer_name,
                                                           descriptor.weights,
                                                           descriptor.biases,
                                                           compute_statistics=True,
                                                           use_mu_sigma=False,
                                                           across_spatial=True,
                                                           input_name=input_name,
                                                           output_name=descriptor.output_names[0])

