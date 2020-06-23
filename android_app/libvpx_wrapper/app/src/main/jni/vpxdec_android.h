//
// Created by hyunho on 6/13/19.
//

#ifndef LIBVPX_WRAPPER_DECODE_TEST_H
#define LIBVPX_WRAPPER_DECODE_TEST_H

#include <vpx/vpx_mobinas.h>

typedef struct vpxdec_cfg_t{
    int arg_skip;
    int threads;
    int stop_after;
    int num_external_frame_buffers;
    int resolution;
    char video_path[PATH_MAX];
    char dnn_path[PATH_MAX];
    char cache_profile_path[PATH_MAX];
} vpxdec_cfg_t;

typedef enum{
    LOW,
    MEDIUM,
    HIGH,
} nemo_dnn_quality;

#ifdef __cplusplus
extern "C" {
#endif

extern int decode(mobinas_cfg_t *, vpxdec_cfg_t *);

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //LIBVPX_WRAPPER_DECODE_TEST_H
