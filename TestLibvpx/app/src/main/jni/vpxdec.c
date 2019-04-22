/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <jni.h>

#include <linux/limits.h>
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <tests/serialize_test.h>

#include "android_example_testlibvpx_MainActivity.h"
#include "./vpx_config.h"
#include "tests/decode_test.h"
//#include "vpx_config.h"

#if CONFIG_LIBYUV
#include "third_party/libyuv/include/libyuv/scale.h"
#endif

#include "./args.h"
#include "./ivfdec.h"

#include "vpx/vpx_decoder.h"
#include "vpx_ports/mem_ops.h"
#include "vpx_ports/vpx_timer.h"

#if CONFIG_VP8_DECODER || CONFIG_VP9_DECODER
#include "vpx/vp8dx.h"
#endif

#include "./md5_utils.h"

#include "./tools_common.h"
#if CONFIG_WEBM_IO
#include "./webmdec.h"
#endif
#include "./y4menc.h"
//#include "vpxdec.h"

#include <android/log.h>
#include <tests/serialize_test.h>

#define TAG "vpxdec.c JNI"
#define _UNKNOWN   0
#define _DEFAULT   1
#define _VERBOSE   2
#define _DEBUG    3
#define _INFO        4
#define _WARN        5
#define _ERROR    6
#define _FATAL    7
#define _SILENT       8
#define LOGUNK(...) __android_log_print(_UNKNOWN,TAG,__VA_ARGS__)
#define LOGDEF(...) __android_log_print(_DEFAULT,TAG,__VA_ARGS__)
#define LOGV(...) __android_log_print(_VERBOSE,TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(_DEBUG,TAG,__VA_ARGS__)
#define LOGI(...) __android_log_print(_INFO,TAG,__VA_ARGS__)
#define LOGW(...) __android_log_print(_WARN,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(_ERROR,TAG,__VA_ARGS__)
#define LOGF(...) __android_log_print(_FATAL,TAG,__VA_ARGS__)
#define LOGS(...) __android_log_print(_SILENT,TAG,__VA_ARGS__)

JNIEXPORT void JNICALL Java_android_example_testlibvpx_MainActivity_vpxDecodeVideo
        (JNIEnv *env, jobject jobj, jstring jstr1, jstring jstr2) //TODO (hyunho): get struct to configure video/test(num_frames, ...) in more details, refer google Keep data
{
    const char *video_dir = (*env)->GetStringUTFChars(env, jstr1, NULL);
    const char *log_dir = (*env)->GetStringUTFChars(env, jstr2, NULL);

    assert(!(define DEBUG_SERIALIZE && define DEBUG_RESIZE));

    video_info_t hr_video_info = {.resolution = 960, .duration=20, .upsample=0, .format="webm", .scale=4};
    video_info_t lr_video_info = {.resolution = 240, .duration=20, .upsample=0, .format="webm", .scale=4};
    video_info_t hr_upsample_video_info = {.resolution = 960, .duration=20, .upsample=1, .format="webm", .scale=4};
    video_info_t lr_video2_info = {.resolution = 288, .duration=11, .upsample=0, .format="webm", .scale=4};


#if DEBUG_SERIALIZE
    decode_test(video_dir, log_dir, hr_video_info);
    decode_test(video_dir, log_dir, lr_video_info);
    decode_test(video_dir, log_dir, hr_upsample_video_info);
#endif

#if DEBUG_RESIZE
    //decode_test(video_dir, log_dir, lr_video2_info);
    decode_test(video_dir, log_dir, lr_video_info); //TODO (hyunho): a) deserialized resized frames, b) save visible frames seperately
    //measure_quality("240p_20", "960p_20", "960p_20_bicubic", log_dir); //TODO: a) measure quality in PSNR/SSIM
#endif

    (*env)->ReleaseStringUTFChars(env, jstr1, video_dir);
    (*env)->ReleaseStringUTFChars(env, jstr2, log_dir);

    LOGI("vpxDecodeVideo ends");
}