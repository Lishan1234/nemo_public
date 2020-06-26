//
// Created by hyunho on 6/13/19.
//

#ifndef LIBVPX_WRAPPER_DECODE_TEST_H
#define LIBVPX_WRAPPER_DECODE_TEST_H

#include <vpx/vpx_nemo.h>

typedef struct vpxdec_cfg_t{
    int arg_skip;
    int threads;
    int stop_after;
    int num_external_frame_buffers;
    int resolution;
    int scale;
    char video_path[PATH_MAX];
    char dnn_path[PATH_MAX];
    char cache_profile_path[PATH_MAX];
} vpxdec_cfg_t;

typedef enum{
    LOW,
    MEDIUM,
    HIGH,
} nemo_dnn_quality;

typedef enum{
    SAVE_REFERENCE_FRAMES,
    MEASURE_SR_LATENCY,
    MEASURE_SR_QUALITY,
    SAVE_SR_FRAMES,
    MEASURE_DECODE_LATENCY,
    MEASURE_DECODE_QUALITY,
    SAVE_DECODE_FRAMES,
    MEASURE_CACHE_LATENCY,
    MEASURE_CACHE_QUALITY,
    SAVE_CACHE_FRAMES,
} test_mode;

#ifdef __cplusplus
extern "C" {
#endif

extern int decode(nemo_cfg_t *, vpxdec_cfg_t *);

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //LIBVPX_WRAPPER_DECODE_TEST_H
