# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from operator import mul
from functools import reduce

from snpe.converters.common.converter_ir import translation, op_adapter
from snpe.converters.common.converter_ir.axis_tracker import AxisTracker
from snpe.converters.common.utils.snpe_converter_utils import *
from snpe.converters.common.utils import code_to_message

# ------------------------------
#   Module Level enum/Functions
# ------------------------------


REMOVE_NOOP = "REMOVE_NOOP"
MATCH_CHANNELSHUFFLE = "MATCH_CHANNELSHUFFLE"
SQUASH_BATCHNORM = "SQUASH_BATCHNORM"
SQUASH_SCALE = "SQUASH_SCALE"
AXES_TO_SPATIAL_FIRST_ORDER = "AXES_TO_SPATIAL_FIRST_ORDER"
supported_opt_list = [SQUASH_SCALE, SQUASH_BATCHNORM, AXES_TO_SPATIAL_FIRST_ORDER, REMOVE_NOOP]


OptimizationTranslations = translation.TranslationBank()


class OptimizationTranslationBase(translation.Translation):
    def __init__(self):
        translation.Translation.__init__(self)
        self.register_method(AXES_TO_SPATIAL_FIRST_ORDER, self.axes_to_spatial_first_order)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


def apply_graph_optimizations(graph, disable_batchnorm_folding=False, perform_axes_to_spatial_first_order=True):

    # apply graph transformations
    OptimizationTranslations.apply_method_to_all_ops(SQUASH_SCALE, graph, fail_if_no_method=False)
    OptimizationTranslations.apply_method_to_all_ops(MATCH_CHANNELSHUFFLE, graph, fail_if_no_method=False)
    if not disable_batchnorm_folding:
        OptimizationTranslations.apply_method_to_all_ops(SQUASH_BATCHNORM, graph, fail_if_no_method=False)

    # transition to NSC
    if perform_axes_to_spatial_first_order:
        OptimizationTranslations.apply_method_to_all_ops(AXES_TO_SPATIAL_FIRST_ORDER, graph)

    # remove NOOPs, which may include trivial permutes at this point
    OptimizationTranslations.apply_method_to_all_ops(REMOVE_NOOP, graph, fail_if_no_method=False)

# ------------------------------------------------------------------------------------------------------------------
#   Translations
#   Note: each Optimization Concrete class has at a minimum 1 optimize function. i.e axes_to_spatial_first_order(..)
#         if more is needed for a given op, it needs to register that method_key and implement a function for it.
# ------------------------------------------------------------------------------------------------------------------


def register(optimization_translation):
    """
    For anything decorated with register in this module, the class along with its op_type is registered in
    a TranslationBank
    :param optimization_translation: a concrete class for a given optimization
    """
    OptimizationTranslations.register_translation(optimization_translation(), optimization_translation().op_type)
    return optimization_translation


@register
class OptimizeInputTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.InputOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if node.op.image_type == 'opaque':
            buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif buf.rank() == 4:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            buf.axis_format = AxisTracker.AxisFormat.NSC
            node.op.shape = buf.shape
        elif buf.rank() == 2:
            buf.axis_format = AxisTracker.AxisFormat.FEATURE
            node.op.shape = buf.shape
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_INPUT_UNEXPECTED_RANK")(node.op.name, buf.rank()))

@register
class OptimizeArgMaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ArgMaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keepdims:
                AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.NCS,
                                                    AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
            axis_map = [0, 3, 1, 2]
            node.op.axis = axis_map[node.op.axis]

@register
class OptimizeBatchnormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BatchnormOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 4:
            AxisTracker.image_to_spatial_first_order(node, graph)
        elif input_buf.rank() == 2 or input_buf.rank() == 3:
            if input_buf.rank() == 3:
                # add custom permute for 3D use-case. This input use-case is added for batchnorm-1D
                AxisTracker.enforce_input_format(graph, node.input_names[0],
                                                 AxisTracker.AxisFormat.NONTRIVIAL, [0, 2, 1])
            output_buf = graph.get_output_buffers(node)[0]
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_BATCHNORM_DIM_UNSUPPORTED")(input_buf.rank()))

    @staticmethod
    def squash_batchnorm(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        prev = input_buffer.producer
        if prev.op.type == 'convolution' and input_buffer.rank() == 4:
            log_debug(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(node.op.name, prev.op.name))

            # The Conv weights are not yet transposed as that happens in axes_to_spatial_first optimization later,
            # so we need to transpose for BN weight broadcasting and then revert
            weights = numpy.transpose(prev.op.weights, (2, 3, 1, 0))
            weights = (weights * node.op.weights)
            prev.op.weights = numpy.transpose(weights, (3, 2, 0, 1))
            prev.op.bias = prev.op.bias * node.op.weights + node.op.bias
            graph.squash(node, input_buffer.name)


@register
class OptimizeChannelShuffleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ChannelShuffleOp.TRANSLATION_KEY

    def axes_to_snpe_order(self, node, graph):
        log_debug(code_to_message.get_debugging_message("DEBUG_AXES_TO_SNPE_ORDER_ENTRY")(node.op.name))
        super(OptimizeChannelShuffleTranslation, self).axes_to_spatial_first_order(node, graph)
        for buf in graph.get_input_buffers(node):
            log_debug("input {} {} {}", buf.name, buf.axis_format, buf.shape)
        for buf in graph.get_output_buffers(node):
            log_debug("output {} {} {}", buf.name, buf.axis_format, buf.shape)


@register
class OptimizeConvolutionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConvolutionOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeConvolutionTranslation, self).axes_to_spatial_first_order(node, graph)
        # if this method is called, current weight order for is NCHW but we want HWCN
        weights = numpy.transpose(node.op.weights, (2, 3, 1, 0))
        node.op.weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)


@register
class OptimizeConcatTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConcatOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NSC:
            axis_map = [0, 3, 1, 2]
            node.op.axis = axis_map[node.op.axis]


@register
class OptimizeConstantTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConstantOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_buffer(node.output_names[0])

        # Permute the constant data if necessary
        # TODO: figure out where these weights are suppose to come from??
        # Code was copied verbatim from data_translations.py during refactoring
        if output_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.tensor = numpy.ascontiguousarray(numpy.transpose(weights, AxisTracker.AxisFormat.NCS_TO_NSC))
        elif output_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.tensor = numpy.ascontiguousarray(numpy.transpose(weights, AxisTracker.AxisFormat.TBF_TO_BTF))

        AxisTracker.eltwise_to_spatial_first_order(node, graph)

    @staticmethod
    def remove_noop(node, graph):
        # Prune this node if it's an input to a weight layer and was used internally
        if graph.weights.consumed(node.output_names[0]):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONSTANT_PRUNED")(node.output_names[0]))
            graph.prune(node)


@register
class OptimizeCropTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CropOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.TBF_TO_BTF)
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register
class OptimizeCrossCorrelationTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CrossCorrelationOp.TRANSLATION_KEY


@register
class OptimizeDeconvolutionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DeconvolutionOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeDeconvolutionTranslation, self).axes_to_spatial_first_order(node, graph)

        # weights are in CNHW, want HWCN
        weights = numpy.transpose(node.op.weights, (2, 3, 0, 1))
        node.op.weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)


@register
class OptimizeElementwiseMaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseMaxOp.TRANSLATION_KEY


@register
class OptimizeElementwiseProductTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseProductOp.TRANSLATION_KEY
        self.register_method(SQUASH_SCALE, self.squash_scale)

    @staticmethod
    def squash_scale(node, graph):
        if hasattr(node.op, 'weights'):
            input_buffer = graph.get_input_buffers(node)[0]
            prev = input_buffer.producer
            log_assert(prev.op.type == op_adapter.BatchnormOp.TRANSLATION_KEY,
                       code_to_message.get_error_message("ERROR_MUL_SCALE_PREV_NOT_BATCHNORM")(prev.op.name,
                                                                                               prev.op.type))
            weights = node.op.weights
            prev.op.weights *= weights
            prev.op.bias *= weights
            graph.squash(node, input_buffer.name)


@register
class OptimizeElementwiseSumTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseSumOp.TRANSLATION_KEY
        self.register_method(SQUASH_SCALE, self.squash_scale)

    @staticmethod
    def squash_scale(node, graph):
        if hasattr(node.op, 'bias'):
            input_buffer = graph.get_input_buffers(node)[0]
            prev = input_buffer.producer
            log_assert(hasattr(prev.op, 'bias'),
                       code_to_message.get_error_message("ERROR_ADD_BIAS_PREV_NO_BIAS")(node.op.name,
                                                                                        prev.op.name,
                                                                                        prev.op.type))
            prev.op.bias += node.op.bias
            graph.squash(node, input_buffer.name)


@register
class OptimizeFullyConnectedTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.FullyConnectedOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.log_axes_to_spatial_first_order(node, graph)
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 4:
            AxisTracker.enforce_input_format(graph, input_buf.name, AxisTracker.AxisFormat.NSC,
                                             AxisTracker.AxisFormat.NCS_TO_NSC)
            # weights expect NCHW order, need to permute
            input_buf = graph.get_input_buffers(node)[0]
            batch, height, width, depth = input_buf.shape
            weights = node.op.weights_list[0]

            # TODO: this optimization was added based on onnx framework. Verify(Modify) if
            #       change is needed for other frameworks
            # ONNX defines FC as W^Tx + b,
            # so the weights have shape (batch, input_size, output_size)
            input_size = weights.shape[0]
            output_size = weights.shape[1]
            log_assert(input_size == depth * height * width,
                       code_to_message.get_error_message("ERROR_FC_WRONG_INPUT_SIZE")(node.op.name,
                                                                                      input_size,
                                                                                      (depth, height, width)))
            weights.shape = (depth, height, width, output_size)
            weights = numpy.transpose(weights, (3, 1, 2, 0))
            weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)
            weights.shape = (output_size, input_size)
            node.op.weights_list[0] = weights
        elif input_buf.rank() == 2:
            # again, need to transpose weights for snpe order
            weights = node.op.weights_list[0]
            weights = numpy.ascontiguousarray(numpy.transpose(weights, (1, 0)))
            node.op.weights_list[0] = weights

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = AxisTracker.AxisFormat.FEATURE


@register
class OptimizeGenerateProposalsOp(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GenerateProposalsOp.TRANSLATION_KEY


@register
class OptimizeGruTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GruOp.TRANSLATION_KEY


@register
class OptimizeLstmTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LstmOp.TRANSLATION_KEY


@register
class OptimizeMaxYTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MaxYOp.TRANSLATION_KEY


@register
class OptimizeNeuronTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NeuronOp.TRANSLATION_KEY


@register
class OptimizeNoopTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.Noop.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_output_buffers(node)[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf.shape = input_buf.shape
        output_buf.axis_format = input_buf.axis_format

    @staticmethod
    def remove_noop(node, graph):
        graph.squash(node, node.input_names[0])


@register
class OptimizePadTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PadOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.pads = AxisTracker.permute_shape(node.op.pads, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.pads = AxisTracker.permute_shape(node.op.pads, AxisTracker.AxisFormat.TBF_TO_BTF)
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register
class OptimizePoolTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PoolOp.TRANSLATION_KEY


@register
class OptimizePermuteTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PermuteOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        # check for trivial cases first, which will end up
        # in removal. Otherwise, just set output order to nontrivial
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            # special case: transforming to NSC, will become noop
            if node.op.order == [0, 2, 3, 1]:
                node.op.order = [0, 1, 2, 3]
                output_buf.axis_format = AxisTracker.AxisFormat.NSC
                return
            else:
                # going to nontrivial
                AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.NCS,
                                                    AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            if node.op.order == [0, 2, 3, 1]:
                node.op.order = [0, 1, 2, 3]
                output_buf.axis_format = AxisTracker.AxisFormat.BTF
            else:
                AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.TBF,
                                                    AxisTracker.AxisFormat.TBF_TO_BTF, [node.op.name])
                output_buf.axis_format = AxisTracker. AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            if len(node.op.order) == 4:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif len(node.op.order) > 4:
                raise ValueError(code_to_message.get_error_message("ERROR_PERMUTE_TOO_MANY_DIMENSIONS")(node.op.order))
            else:
                # nothing to be done
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_PERMUTE_UNEXPECTED_INPUT_ORDER")
                             (input_buf.axis_format))

    @staticmethod
    def remove_noop(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        if input_buffer.axis_format == output_buffer.axis_format and node.op.order == list(range(len(node.op.order))):
            # this permute is trivial, remove it
            graph.squash(node, input_buffer.name)


@register
class OptimizePreluTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PreluOp.TRANSLATION_KEY


@register
class OptimizeProposalTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ProposalOp.TRANSLATION_KEY


@register
class OptimizeReshapeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReshapeOp.TRANSLATION_KEY
        self.register_method(MATCH_CHANNELSHUFFLE, self.match_channelshuffle)

    @staticmethod
    def product(nums):
        if len(nums) == 0:
            return 1
        else:
            return reduce(mul, nums)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        # force convergence if necessary
        # use the 'backwards' permute orders because they are self-inverses.
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.NCS,
                                                AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.TBF,
                                                AxisTracker.AxisFormat.TBF_TO_BTF, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            pass
        elif input_buf.axis_format == AxisTracker.AxisFormat.FEATURE:
            pass
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER")
                             (input_buf.axis_format))

        output_buf = graph.get_output_buffers(node)[0]
        if output_buf.rank() > 4:
            log_assert(self.product(output_buf.shape[:-4]) == 1,
                       code_to_message.get_error_message("ERROR_RESHAPE_BATCH_UNSUPPORTED"))
            output_buf.shape = output_buf.shape[-4:]
        output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

    def match_channelshuffle(self, node, graph):
        first = node

        output_buffer = graph.get_output_buffers(first)[0]
        consumers = output_buffer.consumers.copy()
        if len(consumers) != 1:
            return False
        second = consumers.pop()

        output_buffer = graph.get_output_buffers(second)[0]
        consumers = output_buffer.consumers.copy()
        if len(consumers) != 1:
            return False
        third = consumers.pop()

        is_valid_channelshuffle = self.check_for_channelshuffle(
                                      graph, first, second, third)
        if is_valid_channelshuffle:
            # ChannelShuffle Op found,
            # Squash Permute and 2nd Reshape Op and
            # Replace 1st ReshapeOp with ShuffleOp
            third_input_buffer = graph.get_input_buffers(third)[0]
            graph.squash(third, third_input_buffer.name)

            second_input_buffer = graph.get_input_buffers(second)[0]
            graph.squash(second, second_input_buffer.name)

            output_shape = first.op.output_shape
            # Assuming the shape is N[GC']HW
            groups = output_shape[1]
            shuffle_op = op_adapter.ChannelShuffleOp(
                             None, groups=groups)
            shuffle_op.name = graph.naming_policy.get_op_name(shuffle_op)
            graph.replace(first.op, shuffle_op)

            return True

        return False

    def check_for_channelshuffle(self, graph, first, second, third):
        input_buffer = graph.get_input_buffers(first)[0]
        input_shape = input_buffer.shape
        output_buffer = graph.get_output_buffers(third)[0]
        output_shape = output_buffer.shape

        return (self.check_for_valid_reshape_1(graph, first) and
                self.check_for_valid_permute(second) and
                self.check_for_valid_reshape_2(graph, third) and
                (output_shape == input_shape))

    def check_for_valid_reshape_1(self, graph, node):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        input_shape = input_buffer.shape
        output_shape = output_buffer.shape

        if (len(input_shape) == 4 and
            len(output_shape) == 5 and
            input_shape[0] == output_shape[0] and
            input_shape[2] == output_shape[3] and
            input_shape[3] == output_shape[4]):
                return True

        return False

    def check_for_valid_permute(self, node):
        # Assuming the input shape is N[GC']HW
        if node.op.type == op_adapter.PermuteOp.TRANSLATION_KEY and node.op.order == [0, 2, 1, 3, 4]:
            return True
        return False

    def check_for_valid_reshape_2(self, graph, node):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        input_shape = input_buffer.shape
        output_shape = output_buffer.shape

        if len(input_shape) == 5 and len(output_shape) == 4 and \
           input_shape[0] == output_shape[0] and \
           input_shape[3] == output_shape[2] and \
           input_shape[4] == output_shape[3]:
            return True

        return False


@register
class OptimizeRNormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RNormOp.TRANSLATION_KEY


@register
class OptimizeRoiAlignTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiAlignOp.TRANSLATION_KEY


@register
class OptimizeRoiPoolingTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiPoolingOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.enforce_input_format(graph, node.input_names[0], AxisTracker.AxisFormat.NSC,
                                         AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf = graph.get_output_buffers(node)[0]
        log_assert(output_buf.shape[0] == 1, code_to_message.get_error_message("ERROR_MAX_ROI_POOL_BATCH_UNSUPPORTED"))
        output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf.axis_format = AxisTracker.AxisFormat.NSC


@register
class OptimizeResizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ResizeOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        node.op.output_shape = AxisTracker.permute_shape(node.op.output_shape, AxisTracker.AxisFormat.NCS_TO_NSC)
        AxisTracker.image_to_spatial_first_order(node, graph)


@register
class OptimizeRnnTransformationTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RnnTransformationOp.TRANSLATION_KEY


@register
class OptimizeSliceTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SliceOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.TBF_TO_BTF)
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register
class OptimizeSoftmaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SoftmaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # NB will probably want to switch to 'eltwise' version when we
        # support axis parameter.
        AxisTracker.feature_to_spatial_first_order(node, graph)


@register
class OptimizeStaticTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.StaticOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        pass

    def remove_noop(self, node, graph):
        graph.prune(node)


@register
class OptimizeSubtractMeanTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SubtractMeanOp.TRANSLATION_KEY


@register
class OptimizeUpsampleIndexBaseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UpsampleIndexBasedOp.TRANSLATION_KEY


@register
class OptimizeUpsampleSparseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UpsampleSparseOp.TRANSLATION_KEY
