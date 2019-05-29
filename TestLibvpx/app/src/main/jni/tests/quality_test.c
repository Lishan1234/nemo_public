//
// Created by hyunho on 4/18/19.
//

#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "../vpxdec.h"
#include "../libvpx/vpx_scale/yv12config.h"
#include "../libvpx/vpx_dsp/psnr.h"

#include <android/log.h>
#define TAG "quality_test.c JNI"
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

int quality_test(decode_info_t video_info, const char *log_dir) {
    //iterate: a) stop_num / w/o stop_num, {x}_{y} y++ / x++ 하면서 iterate
    int frame_status = 1;
    int superframe_status = 1;
    int frame_idx = 0;
    int superframe_idx = 0;
    int ret = 0;
    int stop_after = 100;
    char hr_frame_path[PATH_MAX];
    char hr_resize_frame_path[PATH_MAX];
    char hr_upsample_frame_path[PATH_MAX];

    int width = 1702;
    int height = 960;
    int subsampling_x = 1;
    int subsampling_y = 1;
    int byte_alignment = 0;

    YV12_BUFFER_CONFIG *hr_frame = (YV12_BUFFER_CONFIG *) malloc(sizeof(YV12_BUFFER_CONFIG));
    YV12_BUFFER_CONFIG *hr_resize_frame = (YV12_BUFFER_CONFIG *) malloc(sizeof(YV12_BUFFER_CONFIG));
    YV12_BUFFER_CONFIG *hr_upsample_frame = (YV12_BUFFER_CONFIG *) malloc(sizeof(YV12_BUFFER_CONFIG));
    vpx_free_frame_buffer(hr_frame);
    vpx_free_frame_buffer(hr_resize_frame);
    vpx_free_frame_buffer(hr_upsample_frame);

    if (ret = vpx_realloc_frame_buffer(
            hr_frame, width, height, subsampling_x,
            subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
            cm->use_highbitdepth,
#endif
            VP9_DEC_BORDER_IN_PIXELS, byte_alignment,
            NULL, NULL, NULL)) {
        return ret;
    };

    if (ret = vpx_realloc_frame_buffer(
            hr_resize_frame, width, height, subsampling_x,
            subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
            cm->use_highbitdepth,
#endif
            VP9_DEC_BORDER_IN_PIXELS, byte_alignment,
            NULL, NULL, NULL)) {
        return ret;
    };

    if (ret = vpx_realloc_frame_buffer(
            hr_upsample_frame, width, height, subsampling_x,
            subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
            cm->use_highbitdepth,
#endif
            VP9_DEC_BORDER_IN_PIXELS, byte_alignment,
            NULL, NULL, NULL)) {
        return ret;
    };

    //TODO (hyunho): save info for creating vpx_realloc_frame_buffer - current version hardcodes config

    while (frame_idx < stop_after) {
        while (1) {
            memset(hr_frame_path, sizeof(hr_frame_path), 0);
            memset(hr_resize_frame_path, sizeof(hr_frame_path), 0);
            memset(hr_upsample_frame_path, sizeof(hr_frame_path), 0);

            sprintf(hr_frame_path, "%s/%dp_%d_%d_%d", log_dir,
                    video_info.resolution * video_info.scale, video_info.duration, frame_idx,
                    superframe_idx);
            sprintf(hr_resize_frame_path, "%s/%dp_%d_%d_%d", log_dir,
                    video_info.resolution * video_info.scale, video_info.duration, frame_idx,
                    superframe_idx);
            sprintf(hr_upsample_frame_path, "%s/%dp_%d_bicubic_%d_%d", log_dir,
                    video_info.resolution * video_info.scale, video_info.duration, frame_idx,
                    superframe_idx);

            //all files exist
            if ((access(hr_frame_path, F_OK) != -1) && (access(hr_resize_frame_path, F_OK) != -1) &&
                (access(hr_upsample_frame_path, F_OK) != -1)) {

                // 1) file read & deserialize YV12_BUFFER_CONFIG
                FILE *hr_frame_file = fopen(hr_frame_path, "wb");
                if (hr_frame_file == NULL) {
                    LOGE("file open fail: %s", hr_frame_path);
                    return -1;
                }
                FILE *hr_resize_frame_file = fopen(hr_resize_frame_path, "wb");
                if (hr_resize_frame_file == NULL) {
                    LOGE("file open fail: %s", hr_frame_path);
                    return -1;
                }
                FILE *hr_upsample_frame_file = fopen(hr_upsample_frame_path, "wb");
                if (hr_upsample_frame_file == NULL) {
                    LOGE("file open fail: %s", hr_frame_path);
                    return -1;
                }

                if (ret = vpx_deserialize_load(hr_frame, hr_frame_file, width, height,
                                         subsampling_x, subsampling_y, byte_alignment))
                {
                    return ret;
                }
                if (ret = vpx_deserialize_load(hr_resize_frame, hr_resize_frame_file, width, height,
                                               subsampling_x, subsampling_y, byte_alignment))
                {
                    return ret;
                }
                if (ret = vpx_deserialize_load(hr_upsample_frame, hr_upsample_frame_file, width, height,
                                               subsampling_x, subsampling_y, byte_alignment))
                {
                    return ret;
                }

                // 2) measure quality
                PSNR_STATS *psnr_resize = (PSNR_STATS *)malloc(sizeof(PSNR_STATS));
                memset(psnr_resize, sizeof(PSNR_STATS), 0);
                vpx_calc_psnr(hr_frame, hr_resize_frame, psnr_resize);

                PSNR_STATS *psnr_upsample = (PSNR_STATS *)malloc(sizeof(PSNR_STATS));
                memset(psnr_upsample, sizeof(PSNR_STATS), 0);
                vpx_calc_psnr(hr_frame, hr_upsample_frame, psnr_upsample);


                // 3) save log
                //TODO (hyunho): save log

                superframe_idx++;
            }
            //all file don't exist
            else if ((access(hr_frame_path, F_OK) == -1) &&
                       (access(hr_resize_frame_path, F_OK) == -1) &&
                       (access(hr_upsample_frame_path, F_OK) == -1)) {
                break;
            }
            else {
                LOGE("frame file error");
                return -1;
            }
            superframe_idx++;
        }
        frame_idx++;
        superframe_idx = 0;
    }
}