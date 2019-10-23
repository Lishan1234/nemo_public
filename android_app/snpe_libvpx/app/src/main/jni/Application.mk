# Copyright (c) 2017 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

NDK_TOOLCHAIN_VERSION := clang

APP_PLATFORM := android-16
APP_ABI := arm64-v8a
APP_STL := c++_shared
APP_CPPFLAGS += -std=c++11 -fexceptions -frtti
APP_LDFLAGS = -nodefaultlibs -lc -lm -ldl -lgcc

#from libvpx wrapper
APP_OPTIM := release
# APP_STL := gnustl_static
# APP_CPPFLAGS := -frtti
APP_PLATFORM += android-9