//
// Created by hyunho on 6/13/19.
//

#include "libvpx_wrapper.h"

/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#define DEBUG_LATENCY 1 //여기로 옮기자...

#include <jni.h>
#include <linux/limits.h>
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <android/log.h>
#include <iostream>

#include "android_example_testlibvpx_MainActivity.h"

#include <decode_test.h>
#include <map>

#define TAG "libvpx_wrapper.cpp JNI"
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

static void decode_test_hr(vpx_mobinas_cfg_t *mobinas_cfg, const char* hr_video_file)
{
    //name
    strcpy(mobinas_cfg->prefix, hr_video_file);
    strcpy(mobinas_cfg->target_file, hr_video_file);

    //log
    mobinas_cfg->save_serialized_frame = 1;
    mobinas_cfg->save_decoded_frame = 0;
    mobinas_cfg->save_intermediate = 0;
    mobinas_cfg->save_final = 1;
    mobinas_cfg->save_quality_result = 0;
    mobinas_cfg->save_decode_result = 0;

    //mode
    mobinas_cfg->mode = DECODE;

    //adaptive cache
    mobinas_cfg->profile_cache_reset = 0;
    mobinas_cfg->apply_cache_reset = 0;

    decode_test(mobinas_cfg);
}

static void decode_test_lr(vpx_mobinas_cfg_t *mobinas_cfg, const char* lr_video_file)
{
    //name
    strcpy(mobinas_cfg->prefix, lr_video_file);
    strcpy(mobinas_cfg->target_file, lr_video_file);

    //log
    mobinas_cfg->save_serialized_frame = 1;
    mobinas_cfg->save_decoded_frame = 0;
    mobinas_cfg->save_intermediate = 1;
    mobinas_cfg->save_final = 0;
    mobinas_cfg->save_quality_result = 0;
    mobinas_cfg->save_decode_result = 0;

    //mode
    mobinas_cfg->mode = DECODE;

    //adaptive cache
    mobinas_cfg->profile_cache_reset = 0;
    mobinas_cfg->apply_cache_reset = 0;

    decode_test(mobinas_cfg);
}

static void decode_test_sr(vpx_mobinas_cfg_t *mobinas_cfg, const char* hr_video_file, const char* sr_video_file)
{
    //name
    strcpy(mobinas_cfg->prefix, sr_video_file);
    strcpy(mobinas_cfg->target_file, sr_video_file);
    strcpy(mobinas_cfg->compare_file, hr_video_file);

    //log
    mobinas_cfg->save_serialized_frame = 1;
    mobinas_cfg->save_decoded_frame = 0;
    mobinas_cfg->save_intermediate = 1;
    mobinas_cfg->save_final = 0;
    mobinas_cfg->save_quality_result = 0;
    mobinas_cfg->save_decode_result = 0;

    //mode
    mobinas_cfg->mode = DECODE;

    //adaptive cache
    mobinas_cfg->profile_cache_reset = 0;
    mobinas_cfg->apply_cache_reset = 0;

    decode_test(mobinas_cfg);
}

static void decode_test_sr_cache(vpx_mobinas_cfg_t *mobinas_cfg, const char *hr_video_file, const char *lr_video_file, const char *sr_video_file,
                                 int profile_adaptive_cache, int apply_adaptive_cache)
{
    //name
    strcpy(mobinas_cfg->target_file, lr_video_file);
    strcpy(mobinas_cfg->compare_file, hr_video_file);
    strcpy(mobinas_cfg->cache_file, sr_video_file);
    sprintf(mobinas_cfg->prefix, "cache_%s", sr_video_file);

    //log
    mobinas_cfg->save_serialized_frame = 0;
    mobinas_cfg->save_decoded_frame = 0;
    mobinas_cfg->save_intermediate = 0;
    mobinas_cfg->save_final = 0;
    mobinas_cfg->save_quality_result = 0;
    mobinas_cfg->save_decode_result = 1;

    //mode
    mobinas_cfg->mode = DECODE_SR_CACHE;

    //adaptive cache
    mobinas_cfg->profile_cache_reset = profile_adaptive_cache;
    mobinas_cfg->apply_cache_reset = apply_adaptive_cache;

    decode_test(mobinas_cfg);
}

JNIEXPORT void JNICALL Java_android_example_testlibvpx_MainActivity_vpxDecodeVideo
        (JNIEnv *env, jclass jobj, jstring jstr1, jstring jstr2, jstring jstr3, jstring jstr4, jstring jstr5,
         jint jint1,
         jint jint2)
{
    const char *video_dir = env->GetStringUTFChars(jstr1, NULL);
    const char *log_dir = env->GetStringUTFChars(jstr2, NULL);
    const char *frame_dir = env->GetStringUTFChars(jstr3, NULL);
    const char *serialize_dir = env->GetStringUTFChars(jstr4, NULL);
    const char *profile_dir = env->GetStringUTFChars(jstr5, NULL);

    int target_resolution = (int) jint1;

    //initialize mobinas cfg
    vpx_mobinas_cfg_t mobinas_cfg;
    memset(&mobinas_cfg, 0, sizeof(mobinas_cfg));

    strcpy(mobinas_cfg.video_dir, video_dir);
    strcpy(mobinas_cfg.log_dir, log_dir);
    strcpy(mobinas_cfg.serialize_dir, serialize_dir);
    strcpy(mobinas_cfg.frame_dir, frame_dir);
    strcpy(mobinas_cfg.profile_dir, profile_dir);
    mobinas_cfg.target_resolution = target_resolution;
    LOGD("video_dir: %s", mobinas_cfg.video_dir);

    //initialize video file name
    char hr_video_file[PATH_MAX];
    char lr_video_file[PATH_MAX];
    char lr_bicubic_video_file[PATH_MAX];
    char sr_hq_video_file[PATH_MAX];
    char sr_lq_video_file[PATH_MAX];
    char path[PATH_MAX];
    char line[PATH_MAX];
    memset(hr_video_file, 0, sizeof(hr_video_file));
    memset(lr_video_file, 0, sizeof(lr_video_file));
    memset(lr_bicubic_video_file, 0, sizeof(lr_bicubic_video_file));
    memset(sr_hq_video_file, 0, sizeof(sr_hq_video_file));
    memset(sr_lq_video_file, 0, sizeof(sr_lq_video_file));
    memset(path, 0, sizeof(path));
    memset(line, 0, sizeof(line));

    sprintf(path, "%s/video_list", video_dir);
    FILE *file = fopen(path, "r");
    if (file != NULL) {
        int count = 0;
        while (fgets(line, sizeof line, file) != NULL) {
            line[strlen(line) - 1] = 0;
            if (count == 0) {
                sprintf(hr_video_file, "%s", line);
            }
            else if (count == 1) {
                sprintf(lr_video_file, "%s", line);
            }
            else if (count == 2) {
                sprintf(lr_bicubic_video_file, "%s", line);
            }
            else if (count == 3) {
                sprintf(sr_hq_video_file, "%s", line);
            }
            else if (count == 4) {
                sprintf(sr_lq_video_file, "%s", line);
                break;
            }
            count++;
        }

        if(count != 4) {
            LOGE("%s: video list file is corrupted", __func__);
            return;
        }
    } else {
        LOGE("%s: video list file doesn't exists", __func__);
        return;
    }

//    decode_test_hr(&mobinas_cfg, hr_video_file);
//    decode_test_lr(&mobinas_cfg, lr_video_file);
//    decode_test_sr(&mobinas_cfg, hr_video_file, sr_hq_video_file);
//    decode_test_sr_cache(&mobinas_cfg, hr_video_file, lr_video_file, sr_hq_video_file, 1, 0); //profile adaptive cache
    decode_test_sr_cache(&mobinas_cfg, hr_video_file, lr_video_file, sr_hq_video_file, 0, 1); //turn-on adaptive cache
//    decode_test_sr_cache(&mobinas_cfg, hr_video_file, lr_video_file, sr_hq_video_file, 0, 0); //turn-off adaptive cache

    env->ReleaseStringUTFChars(jstr1, video_dir);
    env->ReleaseStringUTFChars(jstr2, log_dir);
    env->ReleaseStringUTFChars(jstr3, frame_dir);
    env->ReleaseStringUTFChars(jstr4, serialize_dir);

    LOGI("vpxDecodeVideo ends");
}