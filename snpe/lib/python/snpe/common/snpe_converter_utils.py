#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2018 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

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
