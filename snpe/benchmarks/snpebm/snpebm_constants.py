#==============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

SNPE_SDK_ROOT = 'SNPE_ROOT'
# for running as a SNPE contributor
ZDL_ROOT = 'ZDL_ROOT'
DIAGVIEW_OPTION = 'DIAGVIEW_OPTION'
SNPE_BENCH_NAME = 'snpe_bench'
SNPE_BENCH_ROOT = 'SNPE_BENCH_ROOT'
SNPE_BENCH_LOG_FORMAT = '%(asctime)s - %(levelname)s - {}: %(message)s'

SNPE_BATCHRUN_EXE = 'snpe-net-run'
SNPE_RUNTIME_LIB = 'libSNPE.so'
SNPE_BATCHRUN_GPU_S_EXE = 'snpe-net-run-g'
SNPE_RUNTIME_GPU_S_LIB = 'libSNPE_G.so'

SNPE_BENCH_SCRIPT = 'snpe-bench_cmds.sh'
SNPE_DLC_INFO_EXE = "snpe-dlc-info"
SNPE_DIAGVIEW_EXE = "snpe-diagview"
DEVICE_TYPE_ARM_ANDROID = 'arm_android'
DEVICE_TYPE_ARM_LINUX = 'arm_linux'
DEVICE_ID_X86 = 'localhost'
ARTIFACT_DIR = "artifacts"
SNPE_BENCH_DIAG_OUTPUT_FILE = "SNPEDiag_0.log"
SNPE_BENCH_DIAG_REMOVE = "SNPEDiag*"
SNPE_BENCH_OUTPUT_DIR_DATETIME_FMT = "%4d-%02d-%02d_%02d:%02d:%02d"
SNPE_BENCH_ANDROID_ARM32_ARTIFACTS_JSON = "snpebm_artifacts.json"
SNPE_BENCH_ANDROID_ARM32_LLVM_ARTIFACTS_JSON = "snpebm_artifacts_android_arm32_llvm.json"
SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS_JSON = "snpebm_artifacts_android_aarch64.json"
SNPE_BENCH_ANDROID_AARCH64_LLVM_ARTIFACTS_JSON = "snpebm_artifacts_android_aarch64_llvm.json"
SNPE_BENCH_LE_ARTIFACTS_JSON = "snpebm_artifacts_le.json"
SNPE_BENCH_LE_GCC48_HF_ARTIFACTS_JSON = "snpebm_artifacts_le_gcc48hf.json"
SNPE_BENCH_LE64_GCC49_ARTIFACTS_JSON = "snpebm_artifacts_le64_gcc49.json"
SNPE_BENCH_LE64_GCC53_ARTIFACTS_JSON = "snpebm_artifacts_le64_gcc53.json"
SNPE_BENCH_LE_OE_GCC64_ARTIFACTS_JSON = "snpebm_artifacts_le_oe_gcc64.json"
SNPE_BENCH_LE64_OE_GCC64_ARTIFACTS_JSON = "snpebm_artifacts_le64_oe_gcc64.json"

CONFIG_DEVICEOSTYPES_ANDROID_ARM32 = 'android'
CONFIG_DEVICEOSTYPES_ANDROID_ARM32_LLVM = 'android-arm32-llvm'
CONFIG_DEVICEOSTYPES_ANDROID_AARCH64 = 'android-aarch64'
CONFIG_DEVICEOSTYPES_ANDROID_AARCH64_LLVM = 'android-aarch64-llvm'
CONFIG_DEVICEOSTYPES_LE = 'le'
CONFIG_DEVICEOSTYPES_LE_GCC48_HF = 'le_gcc4.8hf'
CONFIG_DEVICEOSTYPES_LE64_GCC49 = 'le64_gcc4.9'
CONFIG_DEVICEOSTYPES_LE64_GCC53 = 'le64_gcc5.3'
CONFIG_DEVICEOSTYPES_LE_OE_GCC64 = 'le_oe_gcc6.4'
CONFIG_DEVICEOSTYPES_LE64_OE_GCC64 = 'le64_oe_gcc6.4'

#some values in the JSON fields. used in JSON and in directory creation.
RUNTIMES = {
    'AIP': ' --use_aip',
    'CPU': '',
    'CPU_FXP8': ' --use_fxp_cpu',
    'DSP': ' --use_dsp',
    'GPU': ' --use_gpu',
    'GPU_s': ' --use_gpu',
    'GPU_FP16': ' --use_gpu --gpu_mode float16'
}

RUNTIME_CPU = 'CPU'
RUNTIME_GPU = 'GPU'
RUNTIME_GPU_ONLY = "GPU_s" #This is using GPU standalone libSNPE_G.so, which is built with -Os to optimize for space

UB_MODES = {
    'ub_float': ' --userbuffer_float',
    'ub_tf8': ' --userbuffer_tf8'
}

RUNTIME_UB_MODES = {
    'CPU': ['ub_float'],
    'CPU_FXP8': ['ub_float'],
    'DSP': ['ub_float', 'ub_tf8'],
    'GPU': ['ub_float'],
    'GPU_s': ['ub_float'],
    'GPU_FP16': ['ub_float'],
    'AIP': ['ub_float']
}

ARCH_AARCH64 = "aarch64"
ARCH_ARM = "arm"
ARCH_DSP = "dsp"
ARCH_X86 = "x86"

PLATFORM_OS_LINUX = "linux"
PLATFORM_OS_ANDROID = "android"

COMPILER_GCC64 = "gcc6.4"
COMPILER_GCC53 = "gcc5.3"
COMPILER_GCC49 = "gcc4.9"
COMPILER_GCC48 = "gcc4.8"
COMPILER_CLANG60 = "clang6.0"
STL_GNUSTL_SHARED = "gnustl_shared.so"
STL_LIBCXX_SHARED = "libc++_shared.so"

MEASURE_SNPE_VERSION = 'snpe_version'
MEASURE_TIMING = "timing"
MEASURE_MEM = "mem"

PROFILING_LEVEL_BASIC = "basic"
PROFILING_LEVEL_DETAILED = "detailed"
PROFILING_LEVEL_OFF = "off"

PERF_PROFILE_BALANCED = "balanced"
PERF_PROFILE_DEFAULT = "default"
PERF_PROFILE_POWER_SAVER = "power_saver"
PERF_PROFILE_HIGH_PERFORMANCE = "high_performance"
PERF_PROFILE_SUSTAINED_HIGH_PERFORMANCE = "sustained_high_performance"

LATEST_RESULTS_LINK_NAME = "latest_results"

MEM_LOG_FILE_NAME = "MemLog.txt"

ERRNUM_CONFIG_ERROR = 1
ERRNUM_PARSEARGS_ERROR = 3
ERRNUM_GENERALEXCEPTION_ERROR = 4
ERRNUM_ADBSHELLCMDEXCEPTION_ERROR = 5
ERRNUM_MD5CHECKSUM_FILE_NOT_FOUND_ON_DEVICE = 14
ERRNUM_MD5CHECKSUM_CHECKSUM_MISMATCH = 15
ERRNUM_MD5CHECKSUM_UNKNOWN_ERROR = 16
ERRNUM_NOBENCHMARKRAN_ERROR = 17
