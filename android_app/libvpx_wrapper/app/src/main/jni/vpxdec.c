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
#include <tests/quality_test.h>

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
        (JNIEnv *env, jobject jobj, jstring jstr1, jstring jstr2, jstring jstr3, jstring jstr4, jint jint1, jint jint2) //TODO (hyunho): get struct to configure video/test(num_frames, ...) in more details, refer google Keep data
{
    const char *video_dir = (*env)->GetStringUTFChars(env, jstr1, NULL);
    const char *log_dir = (*env)->GetStringUTFChars(env, jstr2, NULL);
    const char *frame_dir = (*env)->GetStringUTFChars(env, jstr3, NULL);
    const char *serialize_dir = (*env)->GetStringUTFChars(env, jstr4, NULL);

    //TODO (parse): [Python] read multiple log files
    //TODO (save): [Python] a) avg quality - avg latency, b) avg quality (chunk) - chunk idx, c) quality - frame idx, d) latency, #block info, frame index, e) avg latency, variation, f) avg quality, #key frames, #frames, model info

    //TODO (analysis): [C] bilinear interp overhead
    //TODO (optm.): [C] apply loop filter (optional)
    //TODO (anaylsis): [Python] measure SSIM per seconds (optional)
    int target_resolution = (int) jint1;
    int scale = (int) jint2;
    int stop_after = 120;

    //1. Parse video_list
    char path[PATH_MAX];
    memset(path, 0, PATH_MAX);
    sprintf(path, "%s/video_list", video_dir);
    FILE *file = fopen(path, "r");
    decode_info_t decode_info;

    if (file == NULL) LOGD("no file exist: %s", path);
    LOGD("video_dir: %s", video_dir);

    //TODO: allocate video filename
    char hr_video_file[PATH_MAX];
    char lr_video_file[PATH_MAX];
    char lr_bicubic_video_file[PATH_MAX];
    char sr_hq_video_file[PATH_MAX];
    char sr_lq_video_file[PATH_MAX];
    memset(hr_video_file, 0, sizeof(hr_video_file));
    memset(hr_video_file, 0, sizeof(lr_video_file));
    memset(hr_video_file, 0, sizeof(lr_bicubic_video_file));
    memset(hr_video_file, 0, sizeof(sr_hq_video_file));
    memset(hr_video_file, 0, sizeof(sr_lq_video_file));

    //2. Execute evaluation
    //TODO (hyunho):  save intermeidate sr-frames for caching
    int count = 0;
    if (file != NULL) {
        char line[1000];
        while(fgets(line, sizeof line, file) != NULL)
        {
            memset(&decode_info, 0, sizeof(decode_info));
            LOGD("prefix: %s, target_resolution: %d, scale: %d", line, target_resolution, scale);

            //decode HR & save serialize data
            if (count == 0) {
                decode_info.resolution = target_resolution;
                decode_info.upsample = 0;
                decode_info.scale = 4;
                decode_info.save_decoded_frame = 1;
                decode_info.save_serialized_frame = 1;
                decode_info.save_quality_result = 0;
                decode_info.save_decode_result = 0;
                decode_info.save_intermediate = 0;
                decode_info.mode = DECODE;
                decode_info.stop_after = stop_after;

                line[strlen(line) - 1] = 0;
                sprintf(hr_video_file, "%s", line);
                decode_info.target_file = &hr_video_file;
                memset(decode_info.prefix, 0, PATH_MAX);
                sprintf(decode_info.prefix, "%s", hr_video_file);
                decode_test(video_dir, log_dir, frame_dir, serialize_dir, decode_info);
            }
            //decode LR & save serialize data
            else if (count == 1) {
                decode_info.resolution = target_resolution / scale;
                decode_info.upsample = 0;
                decode_info.scale = 4;
                decode_info.save_decoded_frame = 1;
                decode_info.save_serialized_frame = 1;
                decode_info.save_intermediate = 0;
                decode_info.save_quality_result = 0;
                decode_info.save_decode_result = 1;
                decode_info.mode = DECODE;
                decode_info.stop_after = stop_after;

                line[strlen(line) - 1] = 0;
                sprintf(lr_video_file, "%s", line);
                decode_info.target_file = &lr_video_file;
                memset(decode_info.prefix, 0, PATH_MAX);
                sprintf(decode_info.prefix, "%s", lr_video_file);
                decode_test(video_dir, log_dir, frame_dir, serialize_dir, decode_info);
            }
            //decode LR bicubic & save quality
            else if (count == 2) {
                decode_info.resolution = target_resolution;
                decode_info.upsample = 1;
                decode_info.scale = 4;
                decode_info.save_decoded_frame = 0;
                decode_info.save_serialized_frame = 0;
                decode_info.save_intermediate = 0;
                decode_info.save_quality_result = 1;
                decode_info.save_decode_result = 1;
                decode_info.mode = DECODE;
                decode_info.stop_after = stop_after;

                line[strlen(line) - 1] = 0;
                sprintf(lr_bicubic_video_file, "%s", line);
                decode_info.target_file = &lr_bicubic_video_file;
                decode_info.compare_file = &hr_video_file;
                memset(decode_info.prefix, 0, PATH_MAX);
                sprintf(decode_info.prefix, "%s", lr_bicubic_video_file);
                decode_test(video_dir, log_dir, frame_dir, serialize_dir, decode_info);
            }
            //decode SR & save quality
            //decode SR-cache & save quality
            else if (count == 3) {
                decode_info.resolution = target_resolution;
                decode_info.upsample = 0;
                decode_info.scale = 4;
                decode_info.save_decoded_frame = 0;
                decode_info.save_intermediate = 0;
                decode_info.save_serialized_frame = 0;
                decode_info.save_serialized_key_frame = 1;
                decode_info.save_quality_result = 1;
                decode_info.save_decode_result = 1;
                decode_info.mode = DECODE;
                decode_info.stop_after = stop_after;

                line[strlen(line) - 1] = 0;
                sprintf(sr_hq_video_file, "%s", line);
                decode_info.target_file = &sr_hq_video_file;
                decode_info.compare_file = &hr_video_file;
                memset(decode_info.prefix, 0, PATH_MAX);
                sprintf(decode_info.prefix, "%s", sr_hq_video_file);
                decode_test(video_dir, log_dir, frame_dir, serialize_dir, decode_info);

                decode_info.mode = DECODE_CACHE;
                decode_info.resolution = target_resolution / scale;
                decode_info.save_decoded_frame = 1;
                decode_info.save_intermediate = 0;
                decode_info.save_serialized_frame = 0;
                decode_info.save_serialized_key_frame = 0;
                decode_info.save_quality_result = 1;
                decode_info.save_decode_result = 1;
                decode_info.target_file = &lr_video_file;
                decode_info.compare_file = &hr_video_file;
                decode_info.cache_file = &sr_hq_video_file;
                memset(decode_info.prefix, 0, PATH_MAX);
                sprintf(decode_info.prefix, "cache_%s", sr_hq_video_file);
                decode_test(video_dir, log_dir, frame_dir, serialize_dir, decode_info);
            }
            //decode SR & save quality
            else if (count == 4) {
                decode_info.resolution = target_resolution;
                decode_info.upsample = 0;
                decode_info.scale = 4;
                decode_info.save_decoded_frame = 0;
                decode_info.save_serialized_frame = 0;
                decode_info.save_intermediate =  0;
                decode_info.save_quality_result = 1;
                decode_info.save_decode_result = 1;
                decode_info.mode = DECODE;
                decode_info.stop_after = stop_after;

                line[strlen(line) - 1] = 0;
                sprintf(sr_lq_video_file, "%s", line);
                decode_info.target_file = &sr_lq_video_file;
                decode_info.compare_file = &hr_video_file;
                memset(decode_info.prefix, 0, PATH_MAX);
                sprintf(decode_info.prefix, "%s", sr_lq_video_file);
                decode_test(video_dir, log_dir, frame_dir, serialize_dir, decode_info);

                break;
            }
            else {
                LOGE("unexpected video files");
                goto TEST_END;
            }
            count ++;
        }
    }
    else {
        perror(path);
        goto TEST_END;
    }

    if (count != 4) {
        LOGE("not enough video files");
        goto TEST_END;
    }

//    decode_info_t setup_hr_video = {.resolution = 960, .upsample=0, .duration=20, .scale=4, .save_decoded_frame=1, .save_serialized_frame=1,.save_quality_result=0, .mode=DECODE};
//    decode_info_t setup_lr_video = {.resolution = 240, .upsample=0, .duration=20, .scale=4, .save_decoded_frame=1, .save_serialized_frame=1, .save_quality_result=0, .mode=DECODE};
//    decode_info_t setup_hr_upsample_video = {.resolution = 960, .upsample=1, .duration=20, .scale=4, .save_decoded_frame=0, .save_serialized_frame=0, .save_quality_result=1, .mode=DECODE};
//    decode_info_t test_quality_lr_video = {.resolution = 240, .upsample=0, .duration=20, .scale=4, .save_decoded_frame=1, .save_serialized_frame=1, .save_quality_result=1, .mode=DECODE_CACHE};
//    decode_info_t test_runtime_lr_video = {.resolution = 240, .upsample=0, .duration=20, .scale=4, .save_decoded_frame=0, .save_serialized_frame=0, .save_quality_result=0, .mode=DECODE_CACHE};

//    decode_test(video_dir, log_dir, frame_dir, serialize_dir, setup_hr_video);
//    decode_test(video_dir, log_dir, frame_dir, serialize_dir, setup_lr_video);
//    decode_test(video_dir, log_dir, frame_dir, serialize_dir, setup_hr_upsample_video);

//    decode_test(video_dir, log_dir, frame_dir, serialize_dir, test_quality_lr_video);
//    decode_test(video_dir, log_dir, frame_dir, serialize_dir, test_runtime_lr_video);

    TEST_END:

    (*env)->ReleaseStringUTFChars(env, jstr1, video_dir);
    (*env)->ReleaseStringUTFChars(env, jstr2, log_dir);
    (*env)->ReleaseStringUTFChars(env, jstr3, frame_dir);
    (*env)->ReleaseStringUTFChars(env, jstr4, serialize_dir);

    LOGI("vpxDecodeVideo ends");
}