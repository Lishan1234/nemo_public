# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

from snpe.converters.common.converter_ir import translation
from snpe.converters.common.converter_ir import op_adapter


# ------------------------------------------------------------------------------
#   Module Level enum/Functions
# ------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# IR consts to dlc dictionary. This holds the translation between the string constants in IR graph
# to what is defined in modeltools.
# -------------------------------------------------------------------------------------------------
ir_consts_to_dlc = {
    # conv
    "PADDING_ZERO": modeltools.PADDING_ZERO,
    "PADDING_REFLECT": modeltools.PADDING_REFLECT,
    "PADDING_CONSTANT": modeltools.PADDING_CONSTANT,
    "PADDING_SIZE_EXPLICIT": modeltools.PADDING_SIZE_EXPLICIT,
    "PADDING_SIZE_IMPLICIT_VALID": modeltools.PADDING_SIZE_IMPLICIT_VALID,
    "PADDING_SIZE_IMPLICIT_SAME": modeltools.PADDING_SIZE_IMPLICIT_SAME,
    "PADDING_SIZE_EXPLICIT_FLOOR": modeltools.PADDING_SIZE_EXPLICIT_FLOOR,
    "PADDING_SIZE_EXPLICIT_ASYMMETRIC": modeltools.PADDING_SIZE_EXPLICIT_ASYMMETRIC,

    "NEURON_RELU": modeltools.NEURON_RELU,
    "NEURON_RELU_MIN_MAX": modeltools.NEURON_RELU_MIN_MAX,
    "NEURON_TANH": modeltools.NEURON_TANH,
    "NEURON_LOGISTIC": modeltools.NEURON_LOGISTIC,
    "NEURON_ELU": modeltools.NEURON_ELU,
    "NEURON_NONE": modeltools.NEURON_NONE,

    # pooling
    "POOL_MAX": modeltools.POOL_MAX,
    "POOL_AVG": modeltools.POOL_AVG,

    # scaling
    "RESIZE_BILINEAR": modeltools.RESIZE_BILINEAR,
    "RESIZE_NEAREST_NEIGHBOR": modeltools.RESIZE_NEAREST_NEIGHBOR,

    # ssd
    "PRIORBOX_TYPE_CORNER": modeltools.PRIORBOX_TYPE_CORNER,
    "PRIORBOX_TYPE_CENTER_SIZE": modeltools.PRIORBOX_TYPE_CENTER_SIZE,
    "PRIORBOX_TYPE_CORNER_SIZE": modeltools.PRIORBOX_TYPE_CORNER_SIZE,

    # embedding
    "EMBEDDING_PARTITION_STRATEGY_MOD": modeltools.EMBEDDING_PARTITION_STRATEGY_MOD,
    "EMBEDDING_PARTITION_STRATEGY_DIV": modeltools.EMBEDDING_PARTITION_STRATEGY_DIV,

    # channel shuffle
    "CHANNEL_SHUFFLE_GROUPED": modeltools.CHANNEL_SHUFFLE_GROUPED,

    # layer affinity
    "LAYER_AFFINITY_CPU_FLOAT32": modeltools.LAYER_AFFINITY_CPU_FLOAT32,
    "LAYER_AFFINITY_GPU_FLOAT32_16_HYBRID": modeltools.LAYER_AFFINITY_GPU_FLOAT32_16_HYBRID,
    "LAYER_AFFINITY_DSP_FIXED8_TF": modeltools.LAYER_AFFINITY_DSP_FIXED8_TF,
    "LAYER_AFFINITY_GPU_FLOAT16": modeltools.LAYER_AFFINITY_GPU_FLOAT16,
}

# translation method keys
IR_TO_DLC = 'ir_to_dlc'

DlcTranslations = translation.TranslationBank()


def get_dlc_model_from_ir(graph):
    model = modeltools.Model()
    DlcTranslations.apply_method_to_all_ops(IR_TO_DLC, graph, model)
    for buf in graph.list_buffers():
        model.set_buffer_axis_order(buf.name, buf.get_axis_order())
    return model


# ------------------------------------------------------------------------------
#   Translations
# ------------------------------------------------------------------------------
def register(dlc_translation):
    DlcTranslations.register_translation(dlc_translation(), dlc_translation.TARGET)
    return dlc_translation


class DlcTranslationBase(translation.Translation):
    def __init__(self):
        translation.Translation.__init__(self)
        self.register_method(IR_TO_DLC, self.add_ir_node_as_dlc_layer)

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        raise NotImplementedError("add_ir_node_as_dlc_layer for {} not implemented ".
                                  format(str(self.__class__.__name__)))


@register
class DlcInputTranslation(DlcTranslationBase):
    TARGET = op_adapter.InputOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_data_layer(node.op.name,
                             node.op.shape,
                             node.op.image_encoding_in,
                             node.op.image_encoding_out,
                             node.op.image_type)


@register
class DlcArgMaxTranslation(DlcTranslationBase):
    TARGET = op_adapter.ArgMaxOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_argmax_layer(node.op.name,
                               node.input_names[0],
                               node.output_names[0],
                               node.op.axis,
                               node.op.keepdims)


@register
class DlcBatchnormTranslation(DlcTranslationBase):
    TARGET = op_adapter.BatchnormOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_batchnorm_layer(node.op.name,
                                  node.op.weights,
                                  node.op.bias,
                                  node.op.compute_statistics,
                                  node.op.use_mu_sigma,
                                  node.op.across_spatial,
                                  node.input_names[0],
                                  node.output_names[0])


@register
class DlcChannelShuffleTranslation(DlcTranslationBase):
    TARGET = op_adapter.ChannelShuffleOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_channel_shuffle_layer(node.op.name,
                                        node.op.groups,
                                        ir_consts_to_dlc[node.op.shuffle_mode],
                                        node.input_names[0],
                                        node.output_names[0])


@register
class DlcConvolutionTranslation(DlcTranslationBase):
    TARGET = op_adapter.ConvolutionOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_conv_layer(node.op.name,
                             node.op.weights,
                             node.op.bias,
                             node.op.padx,
                             node.op.pady,
                             ir_consts_to_dlc[node.op.padding_mode],
                             ir_consts_to_dlc[node.op.padding_size_strategy],
                             node.op.stridex,
                             node.op.stridey,
                             node.op.dilationx,
                             node.op.dilationy,
                             node.input_names[0],
                             node.output_names[0],
                             node.op.groups)


@register
class DlcConcatTranslation(DlcTranslationBase):
    TARGET = op_adapter.ConcatOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_concatenation_layer(node.op.name,
                                      node.input_names,
                                      node.output_names[0],
                                      node.op.axis)


@register
class DlcConstantTranslation(DlcTranslationBase):
    TARGET = op_adapter.ConstantOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_const_layer(node.op.name,
                              list(node.op.tensor.shape),
                              node.op.tensor)


@register
class DlcCropTranslation(DlcTranslationBase):
    TARGET = op_adapter.CropOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_crop_layer(node.op.name,
                             node.op.offsets,
                             node.op.output_shape,
                             node.input_names[0],
                             node.output_names[0])


@register
class DlcCrossCorrelationTranslation(DlcTranslationBase):
    TARGET = op_adapter.CrossCorrelationOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_cross_correlation_layer(node.op.name,
                                          node.input_names[0],
                                          node.input_names[1],
                                          node.output_names[0])


@register
class DlcDeconvolutionTranslation(DlcTranslationBase):
    TARGET = op_adapter.DeconvolutionOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_deconvolution_layer(node.op.name,
                                      node.op.weights,
                                      node.op.bias,
                                      node.op.stride,
                                      ir_consts_to_dlc[node.op.padding_size_strategy],
                                      ir_consts_to_dlc[node.op.padding],
                                      node.input_names[0],
                                      node.output_names[0],
                                      node.op.output_width,
                                      node.op.output_height,
                                      node.op.groups)


@register
class DlcDropoutTranslation(DlcTranslationBase):
    TARGET = op_adapter.DropoutOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_dropout_layer(node.op.name,
                                node.op.keep,
                                node.input_names[0],
                                node.output_names[0])


@register
class DlcElementwiseMaxTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseMaxOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_max_layer(node.op.name,
                                        node.input_names,
                                        node.output_names[0])


@register
class DlcElementwiseProductTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseProductOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_product_layer(node.op.name,
                                            node.input_names,
                                            node.output_names[0])


@register
class DlcElementwiseSumTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseSumOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        coeffs = node.op.coeffs[:]
        num_missing_coeffs = len(node.input_names) - len(coeffs)
        if num_missing_coeffs > 0:
            coeffs.extend([1.0] * num_missing_coeffs)
        model.add_elementwise_sum_layer(node.op.name,
                                        coeffs,
                                        node.input_names,
                                        node.output_names[0])


@register
class DlcFullyConnectedTranslation(DlcTranslationBase):
    TARGET = op_adapter.FullyConnectedOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_fc_layer(node.op.name,
                           node.op.weights_list,
                           node.op.bias,
                           node.input_names,
                           node.output_names[0])


@register
class DlcGenerateProposalsOp(DlcTranslationBase):
    TARGET = op_adapter.GenerateProposalsOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_generate_proposals_layer(node.op.name,
                                           node.op.spatial_scale,
                                           node.op.pre_nms_top_n,
                                           node.op.post_nms_top_n,
                                           node.op.nms_thresh,
                                           node.op.min_size,
                                           node.op.correct_transform_coords,
                                           node.op.anchors,
                                           node.op.im_info,
                                           node.input_names[0],
                                           node.input_names[1],
                                           node.output_names[0],
                                           node.ouput_names[1])


@register
class DlcGruTranslation(DlcTranslationBase):
    TARGET = op_adapter.GruOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_gru_layer(node.op.name,
                            node.op.state_gate,
                            node.op.forget_gate,
                            node.op.control_gate,
                            ir_consts_to_dlc[node.op.activation],
                            ir_consts_to_dlc[node.op.gate_activation],
                            ir_consts_to_dlc[node.op.rec_gate_activation],
                            node.op.backwards,
                            node.input_names[0],
                            node.output_names[0])


@register
class DlcLstmTranslation(DlcTranslationBase):
    TARGET = op_adapter.LstmOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        input_name = node.input_names[0]
        if len(node.input_names) > 1:
            sequence_continuation_name = node.input_names[1]
        else:
            sequence_continuation_name = ''
        if len(node.input_names) > 3:
            c_0_input_name = node.input_names[-2]
            h_0_input_name = node.input_names[-1]
        else:
            c_0_input_name = ''
            h_0_input_name = ''
        if len(node.input_names) in (3, 5):
            x_static_name = node.input_names[2]
        else:
            x_static_name = ''
        model.add_lstm_layer(node.op.name,
                             node.op.gate_weights,
                             node.op.gate_bias,
                             node.op.recurrent_weights,
                             node.op.w_xc_static,
                             node.op.backward,
                             node.op.reset_state_at_time_step_0,
                             input_name,
                             sequence_continuation_name,
                             x_static_name,
                             c_0_input_name,
                             h_0_input_name,
                             node.output_names)


@register
class DlcMaxYTranslation(DlcTranslationBase):
    TARGET = op_adapter.MaxYOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_max_y_layer(node.op.name,
                              node.input_names[0],
                              node.output_names[0])


@register
class DlcNeuronTranslation(DlcTranslationBase):
    TARGET = op_adapter.NeuronOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_neuron_layer(node.op.name,
                               ir_consts_to_dlc[node.op.neuron_type],
                               node.input_names[0],
                               node.output_names[0],
                               node.op.a,
                               node.op.b,
                               node.op.min_clamp,
                               node.op.max_clamp)


@register
class DlcPadTranslation(DlcTranslationBase):
    TARGET = op_adapter.PadOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_pad_layer(node.op.name,
                            node.input_names[0],
                            node.op.pads,
                            ir_consts_to_dlc[node.op.mode],
                            node.op.constant_value,
                            node.output_names[0])


@register
class DlcPoolTranslation(DlcTranslationBase):
    TARGET = op_adapter.PoolOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_pooling_layer(node.op.name,
                                ir_consts_to_dlc[node.op.pool_type],
                                node.op.size_x,
                                node.op.size_y,
                                node.op.stride_x,
                                node.op.stride_y,
                                node.op.pad_x,
                                node.op.pad_y,
                                ir_consts_to_dlc[node.op.padding_size_strategy],
                                node.input_names[0],
                                node.output_names[0],
                                node.op.pool_region_include_padding)


@register
class DlcPermuteTranslation(DlcTranslationBase):
    TARGET = op_adapter.PermuteOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_permute_layer(node.op.name,
                                node.op.order,
                                node.input_names[0],
                                node.output_names[0])


@register
class DlcPreluTranslation(DlcTranslationBase):
    TARGET = op_adapter.PreluOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_prelu_layer(node.op.name,
                              node.op.coeff.tolist(),
                              node.input_names[0],
                              node.output_names[0])


@register
class DlcProposalTranslation(DlcTranslationBase):
    TARGET = op_adapter.ProposalOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_proposal_layer(node.op.name,
                                 node.op.feat_stride,
                                 node.op.scales,
                                 node.op.ratios,
                                 node.op.anchor_base_size,
                                 node.op.min_bbox_size,
                                 node.op.max_num_proposals,
                                 node.op.max_num_rois,
                                 node.op.iou_threshold_nms,
                                 node.input_names,
                                 node.output_names[0])


@register
class DlcReshapeTranslation(DlcTranslationBase):
    TARGET = op_adapter.ReshapeOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_reshape_layer(node.op.name,
                                node.op.output_shape,
                                node.input_names[0],
                                node.output_names[0])


@register
class DlcRNormTranslation(DlcTranslationBase):
    TARGET = op_adapter.RNormOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        if node.op.across_channels:
            add_method = model.add_cmrn_layer
        else:
            add_method = model.add_local_norm_layer

        add_method(node.op.name,
                   node.op.size,
                   node.op.alpha,
                   node.op.beta,
                   node.op.k,
                   node.input_names[0],
                   node.output_names[0])


@register
class DlcRoiAlignTranslation(DlcTranslationBase):
    TARGET = op_adapter.RoiAlignOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_roialign_layer(node.op.name,
                                 node.op.spatial_scale,
                                 node.op.pooled_size_h,
                                 node.op.pooled_size_w,
                                 node.op.sampling_ratio,
                                 node.input_names[0],
                                 node.input_names[1],
                                 node.output_names[0],
                                 node.output_names[1] if len(node.output_names) > 1 else "",
                                 node.op.tiled_batch_h,
                                 node.op.tiled_batch_w,
                                 node.op.batch_pad_h,
                                 node.op.batch_pad_w,
                                 node.op.pad_value)


@register
class DlcRoiPoolingTranslation(DlcTranslationBase):
    TARGET = op_adapter.RoiPoolingOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_roipooling_layer(node.op.name,
                                   node.op.pooled_size_w,
                                   node.op.pooled_size_h,
                                   node.op.spatial_scale,
                                   node.op.output_shape,
                                   node.input_names,
                                   node.output_names[0])


@register
class DlcResizeTranslation(DlcTranslationBase):
    TARGET = op_adapter.ResizeOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_scaling_layer(node.op.name,
                                node.op.output_shape,
                                node.op.pad_value,
                                node.op.maintain_aspect_ratio,
                                ir_consts_to_dlc[node.op.resize_mode],
                                node.op.scale_height,
                                node.op.scale_width,
                                node.input_names[0],
                                node.output_names[0],
                                node.op.align_corners)


@register
class DlcRnnTransformationTranslation(DlcTranslationBase):
    TARGET = op_adapter.RnnTransformationOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_tx_layer(node.op.name,
                           node.op.weights,
                           node.op.bias,
                           node.op.activation,
                           node.input_names[0],
                           node.output_names[0])


@register
class DlcSliceTranslation(DlcTranslationBase):
    TARGET = op_adapter.SliceOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_slice_layer(node.op.name,
                              node.input_names[0],
                              node.op.axis,
                              node.op.slice_points,
                              node.output_names)


@register
class DlcSoftmaxTranslation(DlcTranslationBase):
    TARGET = op_adapter.SoftmaxOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_softmax_layer(node.op.name,
                                node.input_names[0],
                                node.output_names[0])


@register
class DlcSubtractMeanTranslation(DlcTranslationBase):
    TARGET = op_adapter.SubtractMeanOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_subtract_mean_layer(node.op.name,
                                      node.op.mean_values,
                                      node.input_names[0],
                                      node.output_names[0])


@register
class DlcUpsampleIndexBaseTranslation(DlcTranslationBase):
    TARGET = op_adapter.UpsampleIndexBasedOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        pool_id = model.get_layer_id(node.input_names[1])
        model.add_upsample_layer(node.op.name,
                                 node.op.pool_size,
                                 node.op.pool_stride,
                                 node.op.pad,
                                 node.op.output_height,
                                 node.op.output_width,
                                 node.input_names[0],
                                 node.output_names[0],
                                 pool_id)


@register
class DlcUpsampleSparseTranslation(DlcTranslationBase):
    TARGET = op_adapter.UpsampleSparseOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_upsample_layer(node.op.name,
                                 node.op.pool_size,
                                 node.op.pool_stride,
                                 node.op.pad,
                                 node.op.output_height,
                                 node.op.output_width,
                                 node.input_names[0],
                                 node.output_names[0])
