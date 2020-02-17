
LOCAL_PATH := $(call my-dir)
SNPE_ROOT := $(LOCAL_PATH)/snpe
SNPE_INCLUDE_DIR:= $(SNPE_ROOT)/include/zdl
TARGET_ARCH_ABI := arm64-v8a
APP_STL := c++_shared

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


include $(CLEAR_VARS)
LOCAL_MODULE := libSNPE
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libSNPE.so
LOCAL_EXPORT_C_INCLUDES += $(SNPE_INCLUDE_DIR)
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libSYMPHONYCPU
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libsymphony-cpu.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := snpeJNI
LOCAL_SRC_FILES := jni.cpp CheckRuntime.cpp LoadContainer.cpp LoadInputTensor.cpp SetBuilderOptions.cpp Util.cpp NV21Load.cpp udlExample.cpp CreateUserBuffer.cpp PreprocessInput.cpp SaveOutputTensor.cpp
LOCAL_SHARED_LIBRARIES := libSNPE libSYMPHONYCPU
 LOCAL_LDLIBS := -llog -lGLESv2 -lEGL
include $(BUILD_SHARED_LIBRARY)
