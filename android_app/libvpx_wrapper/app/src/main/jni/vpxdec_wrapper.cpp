//
// Created by hyunho on 6/13/19.
//

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

#include "vpx/vpx_mobinas.h"
#include <vpxdec_android.h>
#include <map>
#include <linux/stat.h>
#include <sys/stat.h>

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

static void _mkdir(const char *dir) {
    char tmp[PATH_MAX];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp),"%s",dir);
    len = strlen(tmp);
    if(tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for(p = tmp + 1; *p; p++)
        if(*p == '/') {
            *p = 0;
            mkdir(tmp, S_IRWXU);
            *p = '/';
        }
    mkdir(tmp, S_IRWXU);
}

static mobinas_cfg setup(const char *content_dir, const char *input_video) {
    mobinas_cfg_t mobinas_cfg = {0};

    sprintf(mobinas_cfg.log_dir, "%s/log/%s", content_dir, input_video);
    sprintf(mobinas_cfg.input_frame_dir, "%s/image/%s", content_dir, input_video);
    _mkdir(mobinas_cfg.log_dir);
    _mkdir(mobinas_cfg.input_frame_dir);

    mobinas_cfg.save_frame = 1;
    mobinas_cfg.decode_mode = DECODE;

    return mobinas_cfg;
}

static mobinas_cfg online_sr(const char *content_dir, const char *input_video, const char *compare_video, const char *dnn_name, const char *dnn_file) {
    mobinas_cfg_t mobinas_cfg = {0};

    sprintf(mobinas_cfg.log_dir, "%s/log/%s/%s", content_dir, input_video, dnn_name);
    sprintf(mobinas_cfg.input_frame_dir, "%s/image/%s", content_dir, input_video);
    sprintf(mobinas_cfg.sr_frame_dir, "%s/image/%s/%s", content_dir, input_video, dnn_name);
    sprintf(mobinas_cfg.sr_compare_frame_dir, "%s/image/%s", content_dir, compare_video);
    _mkdir(mobinas_cfg.log_dir);
    _mkdir(mobinas_cfg.input_frame_dir);
    _mkdir(mobinas_cfg.sr_frame_dir);
    _mkdir(mobinas_cfg.sr_compare_frame_dir);

    mobinas_cfg.save_frame = 1;
    mobinas_cfg.save_quality = 1;
    mobinas_cfg.save_latency = 0;
    mobinas_cfg.save_metadata = 0;

    mobinas_cfg.decode_mode = DECODE_SR;
    mobinas_cfg.dnn_mode = ONLINE_DNN;
    mobinas_cfg.dnn_runtime = GPU_FLOAT32_16_HYBRID;

    sprintf(mobinas_cfg.dnn_path, "%s/checkpoint/%s/%s", content_dir, dnn_name, dnn_file);

    return mobinas_cfg;
}



JNIEXPORT void JNICALL Java_android_example_testlibvpx_MainActivity_vpxdec
        (JNIEnv *env, jclass jobj)
{
    const char *content_dir = "/storage/emulated/0/Android/data/android.example.testlibvpx/files";
    const char *input_video = "240p_s0_d60_encoded.webm";
    const char *compare_video = "960p_s0_d60.webm";
    const char *dnn_name = "EDSR_S_B8_F64_S4";
    const char *dnn_file = "ckpt-100.dlc";

    vpxdec_cfg_t vpxdec_cfg = {0};
    vpxdec_cfg.arg_skip = 0;
    vpxdec_cfg.threads = 1;
    vpxdec_cfg.stop_after = 5;
    vpxdec_cfg.num_external_frame_buffers = 50;

//    sprintf(vpxdec_cfg.video_path, "%s/video/%s", content_dir, compare_video);
//    mobinas_cfg setup_mobinas_cfg = setup(content_dir, compare_video);
//    decode(&setup_mobinas_cfg, &vpxdec_cfg);

    sprintf(vpxdec_cfg.video_path, "%s/video/%s", content_dir, input_video);
    mobinas_cfg online_mobinas_cfg = online_sr(content_dir, input_video, compare_video, dnn_name, dnn_file);
    decode(&online_mobinas_cfg, &vpxdec_cfg);
}