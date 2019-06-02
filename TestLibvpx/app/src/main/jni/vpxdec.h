//
// Created by hyunho on 4/17/19.
//
#include <linux/limits.h>

#ifndef TESTLIBVPX_VPXDEC_H
#define TESTLIBVPX_VPXDEC_H

#endif //TESTLIBVPX_VPXDEC_H

typedef enum{
    DECODE,
    DECODE_SR,
    DECODE_CACHE,
} DECODE_MODE;

typedef struct DecodeInfo{
    char log_dir[PATH_MAX];
    char frame_dir[PATH_MAX];
    char serialize_dir[PATH_MAX];
    int resolution;
    int duration;
    int upsample;
    int scale;
    int save_serialized_frame;
    int save_decoded_frame;
    int save_quality;
    DECODE_MODE mode;
} decode_info_t;
