#
# Copyright (C) 2016 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#LIBWEBM_HEADERS += libvpx/third_party/libwebm/common/mkvmuxer
#libvpx/third_party/libwebm/mkvmuxer \
				   libvpx/third_party/libwebm/common \
				   libvpx/third_party/libwebm/common/mkvmuxer 
#LIBWEBM_COMMON_SRCS += libvpx/third_party/libwebm/common/hdr_util.cc 
#LIBWEBM_MUXER_SRCS += third_party/libwebm/mkvmuxer/mkvmuxer.cc \
                      third_party/libwebm/mkvmuxer/mkvmuxerutil.cc \
                      third_party/libwebm/mkvmuxer/mkvwriter.cc \
#LIBWEBM_PARSER_SRCS = libvpx/third_party/libwebm/mkvparser/mkvparser.cc \
                      libvpx/third_party/libwebm/mkvparser/mkvreader.cc \

WORKING_DIR := $(call my-dir)

include $(CLEAR_VARS)
LIBVPX_ROOT := $(WORKING_DIR)/libvpx
LIBYUV_ROOT := $(WORKING_DIR)/libyuv
MOBINAS_LIB_PATH := $(WORKING_DIR)/../libs/$(TARGET_ARCH_ABI)
$(warning $(MOBINAS_LIB_PATH))

# build libyuv_static.a
#LOCAL_PATH := $(WORKING_DIR)
#LIBYUV_DISABLE_JPEG := "yes"
#include $(LIBYUV_ROOT)/Android.mk

include $(CLEAR_VARS)
LOCAL_MODULE	:= libyuv
LOCAL_SRC_FILES	:= $(MOBINAS_LIB_PATH)/libyuv.so
include $(PREBUILT_SHARED_LIBRARY)

# build libvpx.so
LOCAL_PATH := $(WORKING_DIR)
include libvpx.mk

# build libwebm.so
LOCAL_PATH := $(WORKING_DIR)
include $(LIBVPX_ROOT)/third_party/libwebm/Android.mk

# build libvpxJNI.so
include $(CLEAR_VARS)
LOCAL_PATH := $(WORKING_DIR)
LOCAL_MODULE := libvpxtestJNI
LOCAL_ARM_MODE := arm
LOCAL_CPP_EXTENSION := .cc
LOCAL_SRC_FILES := vpx_test.c
LOCAL_SRC_FILES += libvpx/md5_utils.c
LOCAL_SRC_FILES += libvpx/args.c 
LOCAL_SRC_FILES += libvpx/ivfdec.c 
LOCAL_SRC_FILES += libvpx/tools_common.c
LOCAL_SRC_FILES += libvpx/y4menc.c 
LOCAL_SRC_FILES += libvpx/webmdec.cc 
#LOCAL_LDFLAGS := -L$(MOBINAS_LIB_PATH)
#LOCAL_LDLIBS := -llog -lz -lm -landroid -lyuv
LOCAL_LDLIBS := -llog -lz -lm -landroid 
LOCAL_SHARED_LIBRARIES := libvpx libyuv
LOCAL_STATIC_LIBRARIES := cpufeatures libwebm
include $(BUILD_SHARED_LIBRARY)

$(call import-module,android/cpufeatures)
