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

WORKING_DIR := $(call my-dir)

# build libyuv.so
MOBINAS_LIB_PATH := $(WORKING_DIR)/../libs/$(TARGET_ARCH_ABI)
include $(CLEAR_VARS)
LOCAL_MODULE	:= libyuv
LOCAL_SRC_FILES	:= $(MOBINAS_LIB_PATH)/libyuv.so
include $(PREBUILT_SHARED_LIBRARY)

# build libvpx.so
LOCAL_PATH := $(WORKING_DIR)
include $(LOCAL_PATH)/libvpx.mk

# build libwebm.so
LIBVPX_ROOT := $(WORKING_DIR)/libvpx
LOCAL_PATH := $(WORKING_DIR)
include $(LIBVPX_ROOT)/third_party/libwebm/Android.mk

# build libvpxJNI.so
include $(CLEAR_VARS)
libvpx_test_codes := tests/decode_test.c
LOCAL_PATH := $(WORKING_DIR)
LOCAL_MODULE := libvpxtestJNI
LOCAL_ARM_MODE := arm
LOCAL_CPP_EXTENSION := .cc
LOCAL_SRC_FILES := vpxdec.c
LOCAL_SRC_FILES += libvpx/md5_utils.c
LOCAL_SRC_FILES += libvpx/args.c 
LOCAL_SRC_FILES += libvpx/ivfdec.c 
LOCAL_SRC_FILES += libvpx/tools_common.c
LOCAL_SRC_FILES += libvpx/y4menc.c 
LOCAL_SRC_FILES += libvpx/webmdec.cc
LOCAL_SRC_FILES += $(libvpx_test_codes)
#LOCAL_LDFLAGS := -L$(MOBINAS_LIB_PATH)
#LOCAL_LDLIBS := -llog -lz -lm -landroid -lyuv
LOCAL_LDLIBS := -llog -lz -lm -landroid 
LOCAL_SHARED_LIBRARIES := libvpx libyuv
LOCAL_STATIC_LIBRARIES := cpufeatures libwebm
include $(BUILD_SHARED_LIBRARY)

$(call import-module,android/cpufeatures)
