## Copyright (c) 2011 The WebM project authors. All Rights Reserved.
## 
## Use of this source code is governed by a BSD-style license
## that can be found in the LICENSE file in the root of the source
## tree. An additional intellectual property rights grant can be found
## in the file PATENTS.  All contributing project authors may
## be found in the AUTHORS file in the root of the source tree.
# This file automatically generated by configure. Do not edit!
TOOLCHAIN := armv8-android-gcc
ALL_TARGETS += libs
ALL_TARGETS += tools

PREFIX=/usr/local
ifeq ($(MAKECMDGOALS),dist)
DIST_DIR?=vpx-vp9-nopost-nodocs-armv8-android-v1.7.0-69-g7ae6f9e
else
DIST_DIR?=$(DESTDIR)/usr/local
endif
LIBSUBDIR=lib

VERSION_STRING=v1.7.0-69-g7ae6f9e

VERSION_MAJOR=1
VERSION_MINOR=7
VERSION_PATCH=0

CONFIGURE_ARGS=--force-target=armv8-android-gcc --sdk-path=/home/martin/Desktop/android-ndk-r14b --enable-neon --enable-internal-stats --enable-snpe --disable-examples --disable-docs --enable-realtime-only --disable-vp8 --enable-libyuv --disable-runtime-cpu-detect --disable-internal-stats --extra-cflags=     -isystem /home/martin/Desktop/android-ndk-r14b/sysroot/usr/include/arm-linux-androideabi     -isystem /home/martin/Desktop/android-ndk-r14b/sysroot/usr/include   
CONFIGURE_ARGS?=--force-target=armv8-android-gcc --sdk-path=/home/martin/Desktop/android-ndk-r14b --enable-neon --enable-internal-stats --enable-snpe --disable-examples --disable-docs --enable-realtime-only --disable-vp8 --enable-libyuv --disable-runtime-cpu-detect --disable-internal-stats --extra-cflags=     -isystem /home/martin/Desktop/android-ndk-r14b/sysroot/usr/include/arm-linux-androideabi     -isystem /home/martin/Desktop/android-ndk-r14b/sysroot/usr/include   
