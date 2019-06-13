//
// Created by hyunho on 6/13/19.
//

#ifndef LIBVPX_WRAPPER_DECODE_TEST_H
#define LIBVPX_WRAPPER_DECODE_TEST_H


typedef enum{
    DECODE,
    DECODE_SR,
    LOAD_SR,
    DECODE_CACHE,
} DECODE_MODE;

typedef struct DecodeInfo{
    /*** belows are used for development ***/
    //directory
    const char *video_dir;
    const char *log_dir;
    const char *frame_dir;
    const char *serialize_dir;
    const char *prefix;

    //name
    const char *target_file;
    const char *cache_file;
    const char *compare_file;

    //video
    int resolution;
    int duration;

    //dnn
    int upsample;
    int scale;

    //debug
    int save_serialized_frame;
    int save_serialized_key_frame;
    int save_decoded_frame;
    int save_quality_result;
    int save_decode_result;
    int save_intermediate;
    int stop_after;

    /*** belows are used for test ***/
    //cache
    DECODE_MODE mode;
} decode_info_t;

extern int decode_test(decode_info_t decode_info);

#endif //LIBVPX_WRAPPER_DECODE_TEST_H
