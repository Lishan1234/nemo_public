/* Copyright (c) 2011 The WebM project authors. All Rights Reserved. */
/*  */
/* Use of this source code is governed by a BSD-style license */
/* that can be found in the LICENSE file in the root of the source */
/* tree. An additional intellectual property rights grant can be found */
/* in the file PATENTS.  All contributing project authors may */
/* be found in the AUTHORS file in the root of the source tree. */
#include "vpx/vpx_codec.h"
static const char* const cfg = "--force-target=armv8-android-gcc --enable-neon --enable-internal-stats --disable-examples --disable-docs --enable-realtime-only --disable-vp8 --disable-libyuv --disable-runtime-cpu-detect --enable-external-build --extra-cflags=     -isystem /sysroot/usr/include/arm-linux-androideabi     -isystem /sysroot/usr/include     ";
const char *vpx_codec_build_config(void) {return cfg;}
