# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import argparse
import importlib
import os
import traceback
from google.protobuf import text_format

from snpe.converters.common.utils import snpe_converter_utils, snpe_validation_utils,code_to_message
from snpe.converters.common.converter_ir import op_graph, op_graph_optimizations, translation, op_adapter
from snpe.converters.common.utils.snpe_converter_utils import *
from .caffe_base_translation import CaffeTranslations
from .caffe_policies import CaffeNamePolicy, CaffeShapeInferencePolicy
from .weight_provider import RandomWeightProvider, BlobWeightProvider
import input_translations


# ------------------------------------------------------------------------------
#   The Converter Class
# ------------------------------------------------------------------------------
class CaffeConverter(object):
    def __init__(self):
        self.translations = CaffeTranslations
        self.graph = op_graph.IROpGraph(naming_policy=CaffeNamePolicy(),
                                        shape_inference_policy=CaffeShapeInferencePolicy())
        self.input_model_path = ''
        self.caffe_weights_path = ''
        self.output_model_path = ''
        self.debug = ''
        self.encoding = ''
        self.input_dim = list()
        self.input_types_map = {}
        self.network_dim = []
        self.validation_target = list()
        self.enable_strict_validation = False
        self.disable_batchnorm_folding = False
        self.converter_command = ''
        self.copyright_str = ''
        self.model_version = None
        self.udl_args = []
        self.udl_layer_dict = {}
        self.spec = None
        self.set_options(self.parse_args())

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='Script to convert caffe protobuf configuration into a DLC file.')
        parser._action_groups.pop()

        required = parser.add_argument_group('required arguments')
        required.add_argument('-c', '--caffe_txt', type=str, required=True, help='Input caffe proto txt configuration file')

        optional = parser.add_argument_group('optional arguments')
        optional.add_argument('-b', '--caffe_bin', type=str,
                              help='Input caffe binary file containing the weight data')
        optional.add_argument('-d', '--output_path', type=str,
                              help='Path where the converted Output model should be saved. If not specified, the data '
                                   'will be written to a file with same name as the caffetxt file with a backend extension')
        optional.add_argument('--udl', type=str, nargs=2, metavar=('UDL_MODULE', 'FACTORY_FUNCTION'), default=[],
                              help='Option to add User Defined Layers. Provide Filename, Function name.'
                                   '1.Filename: Name of python module to load for registering custom udl(note: must be '
                                   '            in PYTHONPATH). If file part of package list the package.filename as'
                                   '            you would when doing a python import.'
                                   '2.Function name: Name of the udl factory function that return a dictionary of key'
                                   '                  layer type and value function callback.')
        optional.add_argument('--copyright_file', type=str,
                              help='Path to copyright file. If provided, the content of the file will be added to the output model.')
        optional.add_argument('--encoding', type=str, choices=['argb32', 'rgba', 'nv21', 'bgr'], default='bgr',
                              help='Image encoding of the source images. Default is bgr if not specified')
        optional.add_argument('--model_version', type=str,
                              help='User-defined ASCII string to identify the model, only first 64 bytes will be stored')
        optional.add_argument('--disable_batchnorm_folding', dest="disable_batchnorm_folding", action="store_true",
                              help="If not specified, converter will try to fold batchnorm into previous convolution layer")
        optional.add_argument('--in_layer', type=str, action='append', dest='input_layers',
                              help='Name of the input layer')
        optional.add_argument('--in_type', type=str, choices=['default', 'image', 'opaque'], action='append',
                              dest='input_types', help='Type of data expected by input layer. '
                                                       'Type is default if not specified.')
        optional.add_argument('--validation_target', nargs=2, metavar=('RUNTIME_TARGET', 'PROCESSOR_TARGET'), default=[],
                              action=snpe_validation_utils.ValidateTargetArgs,
                              help="A combination of processor and runtime target against which model will be validated."
                                   "Choices for RUNTIME_TARGET: {cpu, gpu, dsp}."
                                   "Choices for PROCESSOR_TARGET: {snapdragon_801, snapdragon_820, snapdragon_835}."
                                   "If not specified, will validate model against {snapdragon_820, snapdragon_835} across "
                                   "all runtime targets.")
        optional.add_argument('--strict', dest="enable_strict_validation", action="store_true", default=False,
                              help="If specified, will validate in strict mode whereby model will not be produced if it "
                                   "violates constraints of the specified validation target. If not specified, will "
                                   "validate model in permissive mode against the specified validation target.")
        parser.add_argument("--debug", type=int, nargs='?', const=0, default=-1, help="Run the converter in debug mode.")
        optional.add_argument("--verbose", dest="verbose", action="store_true", help="Verbose printing", default=False)

        args = parser.parse_args()

        return args

    def set_options(self, args):
        setup_logging(args)
        self.input_model_path = args.caffe_txt
        self.caffe_weights_path = args.caffe_bin
        self.output_model_path = args.output_path
        self.debug = args.debug
        self.encoding = args.encoding
        self.validation_target = args.validation_target
        self.enable_strict_validation = args.enable_strict_validation
        self.disable_batchnorm_folding = args.disable_batchnorm_folding
        self.converter_command = snpe_converter_utils.sanitize_args(args,
                                                                    args_to_ignore=['model_path', 'm', 'output_path', 'd'])
        self.copyright_str = snpe_converter_utils.get_string_from_txtfile(args.copyright_file)
        self.model_version = args.model_version
        self.udl_args = args.udl

        # Caffe specific:  This control caffe's output with verbose option
        if not snpe_converter_utils.is_log_level_debug():
            # The levels are
            # 0 - debug
            # 1 - info (still a LOT of outputs)
            # 2 - warnings
            # 3 - errors
            os.environ['GLOG_minloglevel'] = '2'

        self.input_types_map = {}
        input_layers = args.input_layers
        input_types = args.input_types
        if input_layers is not None and input_types is not None:
            if len(input_layers) != len(input_types):
                print(code_to_message.get_error_message('ERROR_CAFFE_INPUT_TYPES_LAYER_NAMES_NOT_IN_PAIRS')
                      (input_types, input_layers))
                sys.exit(1)
            self.input_types_map = {input_layers[i]: input_types[i] for i in range(len(input_layers))}

    def convert(self):
        # import of Caffe has to come after the setting of GLOG_minloglevel for it to take effect
        try:
            import caffe
            import caffe.proto.caffe_pb2 as caffe_pb2
        except ImportError:
            raise Exception(code_to_message.get_error_message("ERROR_CAFFE_NOT_FOUND")(str(sys.path)))

        # these need to be imported so they are evaluated and translations are registered.
        # importing them here since the modules import caffe as well.
        from . import data_translations, math_translations, nn_translations, proposal_translation,\
            noop_translations, rnn_translations, udl_translation

        # Add udl module if provided
        if len(self.udl_args):
            log_info("Loading UDLs from module {} using factory function {}", self.udl_args[0], self.udl_args[1])
            udl_factory_func = getattr(importlib.import_module(self.udl_args[0]), self.udl_args[1])
            self.udl_layer_dict = udl_factory_func()
            # standardize the layer types so that type matching is easier below
            self.udl_layer_dict = {converter_type(k, "caffe"): v for k, v in self.udl_layer_dict.items()}

        caffe.set_mode_cpu()
        # get caffe spec
        try:
            self.spec = caffe_pb2.NetParameter()
            with open(self.input_model_path, 'rb') as text_file:
                text_format.Merge(text_file.read(), self.spec)
        except Exception as e:
            print(code_to_message.get_error_message('ERROR_CAFFE_CAFFE_PARSING_ERROR')(self.input_model_path,
                                                                                       str(e)))
            print(code_to_message.get_progress_message('INFO_CAFFE_CAFFE_INSTALLATION_ERROR')(caffe.__file__))
            sys.exit(1)

        # get weight provider
        if self.caffe_weights_path is None:
            self.graph.weights = RandomWeightProvider(self.spec, self.graph)
        else:
            caffe_net = caffe.Net(self.input_model_path, self.caffe_weights_path, caffe.TEST)
            self.graph.weights = BlobWeightProvider(caffe_net.params)

        # Need to add a data layer
        if len(self.spec.input_shape) > 0:
            self.input_dim = list(map(int, self.spec.input_shape[0].dim))
        elif len(self.spec.input_dim) > 0:
            self.input_dim = list(map(int, self.spec.input_dim))

        # TODO: why need to change to 4D? commenting out for now
        # if self.input_dim and len(self.input_dim) != 4:
        #     # input_dim, but not 4-D, add 1 as batch
        #     log_debug("Length of input dim " + str(len(self.input_dim)))
        #     self.input_dim.insert(0, 1)
        if self.input_dim and len(self.spec.input):
            input_type = converter_type("data", "caffe")
            self.translations.apply_method_to_op(input_type, input_translations.ADD_INPUT_OP,
                                                 self.spec.input[0], self.input_dim, input_type, self.graph)

        # If there are additional inputs. create data layers for these. Note that
        # only the input {} input_shape {} syntax is supported here.
        for index in range(1, len(self.spec.input)):
            data_name = str(self.spec.input[index])
            if len(self.spec.input_shape[index].dim) == 4:
                data_dims = list(map(int, self.spec.input_shape[index].dim))
            elif len(self.spec.input_shape[index].dim) == 2:
                # 2-D input_dim. Treat as (batch, depth). TODO: why need to change to 4D? commenting out for now
                # data_dims = list(map(int, [self.spec.input_shape[index].dim[0], self.spec.input_shape[index].dim[1], 1, 1]))
                data_dims = list(map(int, [self.spec.input_shape[index].dim[0], self.spec.input_shape[index].dim[1]]))
            else:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_UNSUPPORTED_INPUT_DIMS')
                                 (str(data_name)))  # TODO: remove restriction if applicable
            input_type = converter_type(self.input_types_map.get(data_name, "default"), "caffe")
            self.translations.apply_method_to_op(input_type, input_translations.ADD_INPUT_OP,
                                                 data_name, data_dims, input_type, self.graph)

        # extract parameters, infer shapes, etc.
        layers = self.spec.layer if len(self.spec.layer) != 0 else self.spec.layers
        for i, layer in enumerate(layers):
            if self._is_in_train_phase(layer):
                # Skip train layers
                continue
            log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, layer.type))
            src_type = converter_type(layer.type, "caffe")

            # cases where input is a layer
            if src_type == converter_type("data", "caffe"):
                if len(self.input_dim) == 0:
                    raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_DATA_LAYER_ERR_NO_INPUT_DIM')
                                     (str(layer.name)))
                self.translations.apply_method_to_op(src_type, input_translations.ADD_INPUT_OP,
                                                     str(layer.top[0]), self.input_dim, src_type, self.graph)

            elif src_type == converter_type("input", "caffe"):
                if not hasattr(layer, "input_param"):
                    raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_NO_INPUT_PARAM_SPECIFIED')
                                     (str(layer.name)))

                input_param = layer.input_param
                self.input_dim = list(map(int, input_param.shape[0].dim))
                self.translations.apply_method_to_op(src_type, input_translations.ADD_INPUT_OP,
                                                     str(layer.top[0]), self.input_dim, src_type, self.graph)
            else:
                try:
                    # first check if layer is UDL. If so call factory function to extract blob info
                    # and then add to graph
                    if src_type in self.udl_layer_dict:
                        udl_obj = self.udl_layer_dict[src_type]
                        udl_layer = (layer, udl_obj)
                        self.translations.apply_method_to_op(op_adapter.UdlOp.TRANSLATION_KEY,
                                                             translation.ADD_OP,
                                                             udl_layer,
                                                             self.graph)
                    else:
                        self.translations.apply_method_to_op(src_type,
                                                             translation.ADD_OP,
                                                             layer,
                                                             self.graph)
                except Exception as e:
                    if snpe_converter_utils.is_log_level_debug():
                        traceback.print_exc()
                    log_error("Node %s: %s" % (layer.name, e))
                    sys.exit(-1)

        return self.graph

    def ir_optimize(self, graph, **kwargs):
        try:
            # apply graph transformations
            op_graph_optimizations.apply_graph_optimizations(graph, self.disable_batchnorm_folding, **kwargs)
            return graph
        except Exception as e:
            if snpe_converter_utils.is_log_level_debug():
                traceback.print_exc()
            log_error(str(e))
            sys.exit(-1)

    @staticmethod
    def _is_in_train_phase(layer):
        if layer.include:
            import caffe.proto.caffe_pb2 as caffe_pb2
            caffe_phases = {pair[0]: pair[1] for pair in list(caffe_pb2.Phase.items())}
            phases = [state.phase for state in layer.include if state.phase is not None]
            return caffe_phases['TRAIN'] in phases
        return False