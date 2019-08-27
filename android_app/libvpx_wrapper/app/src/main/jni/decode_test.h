//
// Created by hyunho on 6/13/19.
//

#ifndef LIBVPX_WRAPPER_DECODE_TEST_H
#define LIBVPX_WRAPPER_DECODE_TEST_H


typedef enum{
    DECODE,
    DECODE_SR,
    LOAD_SR,
    DECODE_SR_CACHE,
    DECODE_BILINEAR,
} DECODE_MODE;

typedef struct DecodeInfo{
    /*** belows are used for development ***/
    //directory
    const char *video_dir;
    const char *log_dir;
    const char *frame_dir;
    const char *serialize_dir;

    //name
    const char *prefix;
    const char *target_file;
    const char *cache_file;
    const char *compare_file;

    //video
    int resolution;
    int duration;

    //dnn
    int scale;

    //log
    int save_serialized_frame;
    int save_decoded_frame;
    int save_intermediate;
    int save_final;
    int save_quality_result;
    int save_decode_result;

    //debug
    int debug_quality;

    //codec
    int stop_after;

    /*** belows are used for test ***/
    //cache
    DECODE_MODE mode;
    int apply_adaptive_cache;
    int apply_sr;
} decode_info_t;

typedef struct frameInfo{
    int current_video_frame;
    int current_super_frame;
} frame_info_t;

#ifdef __cplusplus
extern "C" {
#endif

extern int decode_test(decode_info_t decode_info);

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //LIBVPX_WRAPPER_DECODE_TEST_H
