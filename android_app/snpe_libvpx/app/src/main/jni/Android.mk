LOCAL_PATH := $(call my-dir)

TARGET_ARCH_ABI := arm64-v8a
APP_STL := c++_shared
SNPE_ROOT := $(LOCAL_PATH)/snpe


ifeq ($(TARGET_ARCH_ABI), arm64-v8a)
   ifeq ($(APP_STL), c++_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/aarch64-android-clang6.0
   else
      $(error Unsupported APP_STL: '$(APP_STL)')
   endif
else ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
   ifeq ($(APP_STL), c++_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/arm-android-clang6.0
   else
      $(error Unsupported APP_STL: '$(APP_STL)')
   endif
else
   $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

WORKING_DIR := $(call my-dir)
SNPE_INCLUDE_DIR := $(SNPE_ROOT)/include/zdl
MOBINAS_LIB_PATH :=$(WORKING_DIR)/../libs/$(TARGET_ARCH_ABI)


#libyuv.so
include $(CLEAR_VARS)
LOCAL_MODULE := libyuv
LOCAL_SRC_FILES := $(MOBINAS_LIB_PATH)/libyuv.so
include $(PREBUILT_SHARED_LIBRARY)

#libvpx.so
LOCAL_PATH := $(WORKING_DIR)
include $(LOCAL_PATH)/libvpx.mk

#libwebm.so
LIBVPX_ROOT := $(WORKING_DIR)/libvpx
LOCAL_PATH := $(WORKING_DIR)
include $(LIBVPX_ROOT)/third_party/libwebm/Android.mk

#now build libvpxJNI.so
include $(CLEAR_VARS)
$(warning $(CONFIG_DIR))
LOCAL_PATH := $(WORKING_DIR)
LOCAL_MODULE := libvpxtestJNI
LOCAL_ARM_MODE := arm
LOCAL_CPP_EXTENSION := .cc .cpp
CONFIG_DIR := $(LOCAL_PATH)/libvpx_android_configs/$(TARGET_ARCH_ABI)
LOCAL_C_INCLUDES := $(CONFIG_DIR)
LOCAL_SRC_FILES := libvpx_wrapper.cpp decode_test.c libvpx/md5_utils.c libvpx/args.c libvpx/ivfdec.c libvpx/tools_common.c libvpx/y4menc.c libvpx/webmdec.cc $(libvpx_test_codes)
LOCAL_LDLIBS := -llog -lz -lm -landroid
LOCAL_SHARED_LIBRARIES := libvpx libyuv
LOCAL_STATIC_LIBRARIES := cpufeatures libwebm
include $(BUILD_SHARED_LIBRARY)



include $(CLEAR_VARS)
LOCAL_MODULE := snpeJNI
LOCAL_SRC_FILES := jni.cpp main.cpp CheckRuntime.cpp LoadContainer.cpp LoadInputTensor.cpp SetBuilderOptions.cpp Util.cpp NV21Load.cpp udlExample.cpp CreateUserBuffer.cpp PreprocessInput.cpp SaveOutputTensor.cpp CreateGLBuffer.cpp CreateGLContext.cpp
LOCAL_SHARED_LIBRARIES := libSNPE libSYMPHONYCPU
LOCAL_LDLIBS     := -llog -lGLESv2 -lEGL
include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libSNPE
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libSNPE.so
LOCAL_EXPORT_C_INCLUDES += $(SNPE_INCLUDE_DIR)
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libSYMPHONYCPU
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libsymphony-cpu.so
include $(PREBUILT_SHARED_LIBRARY)

$(call import-module,android/cpufeatures)

