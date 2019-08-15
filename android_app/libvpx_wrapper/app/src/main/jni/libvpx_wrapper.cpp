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

//TODO: refactor reading video file names, setup, run_cache, ...
int static setup(decode_info_t decode_info, const char *path) {
    FILE *file = fopen(path, "r");

    char hr_video_file[PATH_MAX];
    char lr_video_file[PATH_MAX];
    char lr_bicubic_video_file[PATH_MAX];
    char sr_hq_video_file[PATH_MAX];
    char sr_lq_video_file[PATH_MAX];
    memset(hr_video_file, 0, sizeof(hr_video_file));
    memset(lr_video_file, 0, sizeof(lr_video_file));
    memset(lr_bicubic_video_file, 0, sizeof(lr_bicubic_video_file));
    memset(sr_hq_video_file, 0, sizeof(sr_hq_video_file));
    memset(sr_lq_video_file, 0, sizeof(sr_lq_video_file));

    int count = 0;
    if (file != NULL) {
        char line[1000];
        while (fgets(line, sizeof line, file) != NULL) {
            //LOGD("prefix: %s, target_resolution: %d, scale: %d", line, target_resolution, scale);
            //decode HR & save serialize data
            line[strlen(line) - 1] = 0;
            if (count == 0) {
                LOGD("HR decode start");
                //mode
                decode_info.mode = DECODE;

                //log
                decode_info.save_serialized_frame = 1;
                decode_info.save_decoded_frame = 0;
                decode_info.save_intermediate = 1;
                decode_info.save_final = 1;
                decode_info.save_quality_result = 0;
                decode_info.save_decode_result = 0;

                //name
                sprintf(hr_video_file, "%s", line);
                decode_info.target_file = hr_video_file;
                decode_info.prefix = hr_video_file;
//                decode_test(decode_info);
                LOGD("HR decode end");
            }
                //decode LR & save serialize data
            else if (count == 1) {
                LOGD("LR decode start");
                //mode
                decode_info.mode = DECODE;

                //log
//                decode_info.save_serialized_frame = 1;
//                decode_info.save_decoded_frame = 1;
//                decode_info.save_intermediate = 1;
//                decode_info.save_final = 0;
//                decode_info.save_quality_result = 0;
//                decode_info.save_decode_result = 1;
                decode_info.save_serialized_frame = 0;
                decode_info.save_decoded_frame = 0;
                decode_info.save_intermediate = 0;
                decode_info.save_final = 0;
                decode_info.save_quality_result = 0;
                decode_info.save_decode_result = 0;

                //name
                sprintf(lr_video_file, "%s", line);
                decode_info.target_file = lr_video_file;
                decode_info.prefix = lr_video_file;
                decode_test(decode_info);
                LOGD("LR decode end");
            }
                //decode LR bicubic & save quality
            else if (count == 2) {
                LOGD("LR bicubic decode start");
                //mode
                decode_info.mode = DECODE;

                //log
                decode_info.save_serialized_frame = 0;
                decode_info.save_decoded_frame = 0;
                decode_info.save_intermediate = 0;
                decode_info.save_final = 0;
                decode_info.save_quality_result = 1;
                decode_info.save_decode_result = 1;

                //name
                sprintf(lr_bicubic_video_file, "%s", line);
                decode_info.target_file = lr_bicubic_video_file;
                decode_info.compare_file = hr_video_file;
                decode_info.prefix = lr_bicubic_video_file;
//                decode_test(decode_info);
                LOGD("LR bicubic decode end");
            }
            //decode SR & save quality
            else if (count == 3) {
                LOGD("hqSR decode start");
                //mode
                decode_info.mode = DECODE;

                //log
                decode_info.save_serialized_frame = 1;
                decode_info.save_decoded_frame = 1;
                decode_info.save_intermediate = 1;
                decode_info.save_final = 0;
                decode_info.save_quality_result = 1;
                decode_info.save_decode_result = 1;

                //name
                sprintf(sr_hq_video_file, "%s", line);
                decode_info.target_file = sr_hq_video_file;
                decode_info.compare_file = hr_video_file;
                decode_info.prefix = sr_hq_video_file;
//                decode_test(decode_info);
                LOGD("hqSR decode end");
            }
                //decode SR & save quality
            else if (count == 4) {
                LOGD("lqSR decode start");
                //mode
                decode_info.mode = DECODE;

                //log
                decode_info.save_serialized_frame = 0;
                decode_info.save_decoded_frame = 0;
                decode_info.save_intermediate = 0;
                decode_info.save_final = 0;
                decode_info.save_quality_result = 1;
                decode_info.save_decode_result = 1;

                //name
                sprintf(sr_lq_video_file, "%s", line);
                decode_info.target_file = sr_lq_video_file;
                decode_info.compare_file = hr_video_file;
                decode_info.prefix = sr_lq_video_file;
//                decode_test(decode_info);
                LOGD("lqSR decode end");
                break;
            } else {
                LOGE("unexpected video files");
                return 1;
            }
            count++;
        }
    } else {
        perror(path);
        return 1;
    }

    if (count != 4) {
        LOGE("not enough video files");
        return 1;
    }

    return 0;
}

int static run_cache(decode_info_t decode_info, const char *path) {
    FILE *file = fopen(path, "r");

    char hr_video_file[PATH_MAX];
    char lr_video_file[PATH_MAX];
    char lr_bicubic_video_file[PATH_MAX];
    char sr_hq_video_file[PATH_MAX];
    char sr_cache_hq_video_file[PATH_MAX];
    char sr_lq_video_file[PATH_MAX];
    memset(hr_video_file, 0, sizeof(hr_video_file));
    memset(lr_video_file, 0, sizeof(lr_video_file));
    memset(lr_bicubic_video_file, 0, sizeof(lr_bicubic_video_file));
    memset(sr_hq_video_file, 0, sizeof(sr_hq_video_file));
    memset(sr_cache_hq_video_file, 0, sizeof(sr_cache_hq_video_file));
    memset(sr_lq_video_file, 0, sizeof(sr_lq_video_file));

    //setup
    int count = 0;
    if (file != NULL) {
        char line[1000];
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
                LOGD("hqSR cache decode start");
                //cache
                decode_info.mode = DECODE_CACHE;
                decode_info.apply_adaptive_cache = 1;

                //log
                decode_info.save_serialized_frame = 0;
                decode_info.save_decoded_frame = 1;
                decode_info.save_intermediate = 1;
                decode_info.save_final = 0;
                decode_info.save_quality_result = 1;
                decode_info.save_decode_result = 1;

                //name
                sprintf(sr_hq_video_file, "%s", line);
                decode_info.target_file = lr_video_file;
                decode_info.compare_file = hr_video_file;
                decode_info.cache_file = sr_hq_video_file;
                sprintf(sr_cache_hq_video_file, "cache_%s", sr_hq_video_file);
                decode_info.prefix = sr_cache_hq_video_file;
                decode_test(decode_info);
                LOGD("hqSR cache decode end");
            }
            else if (count == 4) {
                sprintf(sr_lq_video_file, "%s", line);
                break;
            } else {
                LOGE("unexpected video files");
                return 1;
            }
            count++;
        }
    } else {
        perror(path);
        return 1;
    }

    if (count != 4) {
        LOGE("not enough video files");
        return 1;
    }

    return 0;
}

int static run_bilinear(decode_info_t decode_info, const char *path) {
    FILE *file = fopen(path, "r");

    char hr_video_file[PATH_MAX];
    char lr_video_file[PATH_MAX];
    char lr_bicubic_video_file[PATH_MAX];
    char sr_hq_video_file[PATH_MAX];
    char sr_cache_hq_video_file[PATH_MAX];
    char sr_lq_video_file[PATH_MAX];
    memset(hr_video_file, 0, sizeof(hr_video_file));
    memset(lr_video_file, 0, sizeof(lr_video_file));
    memset(lr_bicubic_video_file, 0, sizeof(lr_bicubic_video_file));
    memset(sr_hq_video_file, 0, sizeof(sr_hq_video_file));
    memset(sr_cache_hq_video_file, 0, sizeof(sr_cache_hq_video_file));
    memset(sr_lq_video_file, 0, sizeof(sr_lq_video_file));

    //setup
    int count = 0;
    if (file != NULL) {
        char line[1000];
        while (fgets(line, sizeof line, file) != NULL) {
            line[strlen(line) - 1] = 0;
            if (count == 0) {
                sprintf(hr_video_file, "%s", line);
            }
            else if (count == 1) {
                //cache
                decode_info.mode = DECODE_BILINEAR;
                decode_info.apply_adaptive_cache = 0;

                //log
                decode_info.save_serialized_frame = 0;
                decode_info.save_decoded_frame = 0;
                decode_info.save_intermediate = 0;
                decode_info.save_final = 0;
                decode_info.save_quality_result = 0;
                decode_info.save_decode_result = 1;

                //name
                sprintf(lr_video_file, "%s", line);
                decode_info.target_file = lr_video_file;
                decode_info.compare_file = hr_video_file;
                decode_info.prefix = lr_video_file;
                decode_test(decode_info);
                break;
            }
            else {
                LOGE("unexpected video files");
                return 1;
            }
            count++;
        }
    } else {
        perror(path);
        return 1;
    }

    return 0;
}

inline bool exists_test3 (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

JNIEXPORT void JNICALL Java_android_example_testlibvpx_MainActivity_vpxDecodeVideo
        (JNIEnv *env, jclass jobj, jstring jstr1, jstring jstr2, jstring jstr3, jstring jstr4,
         jint jint1,
         jint jint2)
{
    const char *video_dir = env->GetStringUTFChars(jstr1, NULL);
    const char *log_dir = env->GetStringUTFChars(jstr2, NULL);
    const char *frame_dir = env->GetStringUTFChars(jstr3, NULL);
    const char *serialize_dir = env->GetStringUTFChars(jstr4, NULL);

    int target_resolution = (int) jint1;
    int scale = (int) jint2;
    int stop_after = 60;

    decode_info_t decode_info;

    decode_info.video_dir = video_dir;
    decode_info.log_dir = log_dir;
    decode_info.serialize_dir = serialize_dir;
    decode_info.frame_dir = frame_dir;
    decode_info.resolution = target_resolution;
    decode_info.scale = scale;
    decode_info.stop_after = stop_after;

    char path[PATH_MAX];
    memset(path, 0, PATH_MAX);
    sprintf(path, "%s/video_list", video_dir);

    //setup
    if (setup(decode_info, path))
    {
        LOGD("setup failed");
        return;
    }

    //run bilienar
//    if (run_bilinear(decode_info, path))
//    {
//        LOGD("run cache failed");
//        return;
//    }

    //run cache
    if (run_cache(decode_info, path))
    {
        LOGD("run cache failed");
        return;
    }

    env->ReleaseStringUTFChars(jstr1, video_dir);
    env->ReleaseStringUTFChars(jstr2, log_dir);
    env->ReleaseStringUTFChars(jstr3, frame_dir);
    env->ReleaseStringUTFChars(jstr4, serialize_dir);

    LOGI("vpxDecodeVideo ends");
}