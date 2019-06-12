#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy
import sys

import logging

logger = logging.getLogger(__name__)


def sanitize_args(args, args_to_ignore=[]):
    sanitized_args = []
    for k, v in list(vars(args).items()):
        if k in args_to_ignore:
            continue
        sanitized_args.append('{}={}'.format(k, v))
    return "{} {}".format(sys.argv[0].split('/')[-1], ' '.join(sanitized_args))


def get_string_from_txtfile(filename):
    if not filename:
        return filename
    if filename.endswith('.txt'):
        try:
            with open(filename, 'r') as myfile:
                file_data = myfile.read()
            return file_data
        except Exception as e:
            logger.error("Unable to open file %s: %s" % (filename, e))
            sys.exit(-1)
    else:
        logger.error("File %s: must be a text file." % filename)
        sys.exit(-1)


# @deprecated
# TODO: remove once cleanup of converters is done to use method below instead
def setUpLogger(verbose):
    formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
    lvl = logging.INFO
    if verbose:
         lvl = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(lvl)
    formatter = logging.Formatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class SNPEUtils(object):
    def blob2arr(self, blob):
        if hasattr(blob, "shape"):
            return numpy.ndarray(buffer=blob.data, shape=blob.shape, dtype=numpy.float32)
        else:
            # Caffe-Segnet fork doesn't have shape field exposed on blob.
            return numpy.ndarray(buffer=blob.data, shape=blob.data.shape, dtype=numpy.float32)


# -----------
#   Logging
# -----------
LOGGER = None


def setup_logging(args):
    global LOGGER

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    LOGGER.addHandler(handler)


def log_assert(cond, msg, *args):
    assert cond, msg.format(*args)


def log_debug(msg, *args):
    if LOGGER:
        LOGGER.debug(msg.format(*args))


def log_error(msg, *args):
    if LOGGER:
        LOGGER.error(msg.format(*args))


def log_info(msg, *args):
    if LOGGER:
        LOGGER.info(msg.format(*args))


def log_warning(msg, *args):
    if LOGGER:
        LOGGER.warning(msg.format(*args))
