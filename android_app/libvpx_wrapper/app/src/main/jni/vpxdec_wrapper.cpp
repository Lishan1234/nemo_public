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

#include "vpx/vpx_nemo.h"
#include <vpxdec_android.h>
#include <map>
#include <linux/stat.h>
#include <sys/stat.h>
#include <vpx_mem/vpx_mem.h>
#include <cmath>

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


const char *root_dir = "/storage/emulated/0/Android/data/android.example.testlibvpx/files";

static const char *get_video_name(int resolution) {
    if (resolution == 240) return "240p_512kbps_s0_d300.webm";
    else if (resolution == 360) return "360p_1024kbps_s0_d300.webm";
    else if (resolution == 480) return "480p_1600kbps_s0_d300.webm";
    else if (resolution == 720) return "720p_2640kbps_s0_d300.webm";
    else if (resolution == 1080) return "1080p_4400kbps_s0_d300.webm";
//    else if (resolution == 1440) return "1440p_1600kbps_s0_d300.webm";
    else if (resolution == 2160) return "2160p_s0_d300.webm";
    else return NULL;
}

static const char *get_dnn_name(int resolution, nemo_dnn_quality quality) {
    if (resolution == 240) {
        if (quality == LOW) return "NEMO_S_B8_F9_S4_deconv";
        else if (quality == MEDIUM) return "NEMO_S_B8_F21_S4_deconv";
        else if (quality == HIGH) return "NEMO_S_B8_F32_S4_deconv";
    }
    else if (resolution == 360) {
        if (quality == LOW) return "NEMO_S_B4_F8_S3_deconv";
        else if (quality == MEDIUM) return "NEMO_S_B4_F18_S3_deconv";
        else if (quality == HIGH) return "NEMO_S_B4_F29_S3_deconv";
    }
    else if (resolution == 480) {
        if (quality == LOW) return "NEMO_S_B4_F4_S2_deconv";
        else if (quality == MEDIUM) return "NEMO_S_B4_F9_S2_deconv";
        else if (quality == HIGH) return "NEMO_S_B4_F18_S2_deconv";
    }
    else
        return NULL;
}

static void setup_directory(nemo_cfg_t *nemo_cfg, const char *content, int resolution, nemo_dnn_quality quality, nemo_decode_mode decode_mode) {
    const char *input_video_name = get_video_name(resolution);
    const char *reference_video_name = get_video_name(2160);
    const char *dnn_name = get_dnn_name(resolution, quality);

    if (decode_mode == DECODE) {
        sprintf(nemo_cfg->log_dir, "%s/%s/log/%s", root_dir, content, input_video_name);
        sprintf(nemo_cfg->input_frame_dir, "%s/%s/image/%s", root_dir, content, input_video_name);
        sprintf(nemo_cfg->input_reference_frame_dir, "%s/%s/image/%s", root_dir, content, reference_video_name);
    }
    else if (decode_mode == DECODE_SR || decode_mode == DECODE_CACHE) {
        sprintf(nemo_cfg->log_dir, "%s/%s/log/%s", root_dir, content, input_video_name, dnn_name);
        sprintf(nemo_cfg->input_frame_dir, "%s/%s/image/%s", root_dir, content, input_video_name);
        sprintf(nemo_cfg->sr_frame_dir, "%s/%s/image/%s/%s", root_dir, content, input_video_name, dnn_name);
        sprintf(nemo_cfg->input_reference_frame_dir, "%s/%s/image/%s", root_dir, content, input_video_name);
        sprintf(nemo_cfg->sr_reference_frame_dir, "%s/%s/image/%s", root_dir, content, reference_video_name);
        sprintf(nemo_cfg->sr_offline_frame_dir, "%s/%s/image/%s/%s", root_dir, content, reference_video_name, dnn_name);
    }
}

static void setup_mode_and_log(nemo_cfg_t *nemo_cfg, vpxdec_cfg_t *vpxdec_cfg, test_mode mode, int input_resolution, int reference_resolution, int target_width, int target_height) {
    nemo_cfg->target_width = target_width;
    nemo_cfg->target_height = target_height;
    nemo_cfg->filter_interval = 0;
    nemo_cfg->save_metadata = 1;

    switch (mode) {
        case SAVE_REFERENCE_FRAMES:
            nemo_cfg->decode_mode = DECODE;
            nemo_cfg->save_rgbframe = 1;
            vpxdec_cfg->resolution = reference_resolution;
            break;
        case MEASURE_DECODE_LATENCY:
            nemo_cfg->decode_mode = DECODE;
            nemo_cfg->save_latency = 1;
            vpxdec_cfg->resolution = input_resolution;
            break;
        case MEASURE_DECODE_QUALITY:
            nemo_cfg->decode_mode = DECODE;
            nemo_cfg->save_quality = 1;
            vpxdec_cfg->resolution = input_resolution;
            break;
        case SAVE_DECODE_FRAMES:
            nemo_cfg->decode_mode = DECODE;
            nemo_cfg->save_rgbframe = 1;
            vpxdec_cfg->resolution = input_resolution;
            break;
        case MEASURE_SR_LATENCY:
            nemo_cfg->decode_mode = DECODE_SR;
            nemo_cfg->dnn_mode = ONLINE_DNN;
            nemo_cfg->dnn_runtime = GPU_FLOAT16;
            nemo_cfg->save_latency = 1;
            vpxdec_cfg->resolution = input_resolution;
            break;
        case MEASURE_SR_QUALITY:
            nemo_cfg->decode_mode = DECODE_SR;
            nemo_cfg->dnn_mode = ONLINE_DNN;
            nemo_cfg->dnn_runtime = GPU_FLOAT16;
            nemo_cfg->save_quality = 1;
            vpxdec_cfg->resolution = input_resolution;
            break;
        case SAVE_SR_FRAMES:
            nemo_cfg->decode_mode = DECODE_SR;
            nemo_cfg->dnn_mode = ONLINE_DNN;
            nemo_cfg->dnn_runtime = GPU_FLOAT16;
            nemo_cfg->save_rgbframe = 1;
            vpxdec_cfg->resolution = input_resolution;
            break;
    }
}

JNIEXPORT void JNICALL Java_android_example_testlibvpx_MainActivity_vpxdec
        (JNIEnv *env, jclass jobj)
{
    /* setup a test target */
    const char *content = "product_review1";
    nemo_dnn_quality quality = HIGH;
    int input_resolution = 240;
    int reference_resolution = 2160;
    int target_height = 1080;
    int target_width = 1920;
//    test_mode mode = SAVE_REFERENCE_FRAMES;
//    test_mode mode = MEASURE_DECODE_QUALITY;
    test_mode mode = SAVE_DECODE_FRAMES;
//    test_mode mode = MEASURE_SR_QUALITY;
//    test_mode mode = SAVE_SR_FRAMES;

    /* setup nemo_cfg, vpxdec_cdf */
    nemo_cfg_t *nemo_cfg = init_nemo_cfg();
    vpxdec_cfg_t vpxdec_cfg = {0};
    setup_mode_and_log(nemo_cfg, &vpxdec_cfg, mode, input_resolution, reference_resolution, target_width, target_height);
    setup_directory(nemo_cfg, content, vpxdec_cfg.resolution, quality, nemo_cfg->decode_mode);
    vpxdec_cfg.arg_skip = 0;
    vpxdec_cfg.threads = 1;
    vpxdec_cfg.stop_after = 10;
    vpxdec_cfg.num_external_frame_buffers = 50;
    vpxdec_cfg.scale = (int) floor (nemo_cfg->target_height / vpxdec_cfg.resolution);
//    sprintf(vpxdec_cfg.video_path, "%s/%s/video/%s", root_dir, content, get_video_name(vpxdec_cfg.resolution));
    sprintf(vpxdec_cfg.video_path, "%s/240p_ver2.webm", root_dir);
    sprintf(vpxdec_cfg.dnn_path, "%s/%s/checkpoint/%s/%s.dlc", root_dir, content, get_video_name(vpxdec_cfg.resolution), get_dnn_name(vpxdec_cfg.resolution, quality));
//    sprintf(vpxdec_cfg.dnn_path, "%s/%s/video/%s", root_dir, content, get_video_name(resolution)); //TODO: cache profile path

    /* debug */
//    LOGD("log_dir: %s", nemo_cfg->log_dir);
//    LOGD("input frame_dir: %s", nemo_cfg->input_frame_dir);
//    LOGD("sr frame_dir: %s", nemo_cfg->sr_frame_dir);
//    LOGD("input reference frame_dir: %s", nemo_cfg->input_reference_frame_dir);
//    LOGD("sr reference frame_dir: %s", nemo_cfg->sr_reference_frame_dir);
//    LOGD("sr offline frame_dir: %s", nemo_cfg->sr_offline_frame_dir);

    decode(nemo_cfg, &vpxdec_cfg);
    remove_nemo_cfg(nemo_cfg);
}