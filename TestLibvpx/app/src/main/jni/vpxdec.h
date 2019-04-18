//
// Created by hyunho on 4/17/19.
//

#ifndef TESTLIBVPX_VPXDEC_H
#define TESTLIBVPX_VPXDEC_H

#endif //TESTLIBVPX_VPXDEC_H

typedef struct VideoInfo{
    int resolution;
    int duration;
    int upsample;
    char format[PATH_MAX];
    int scale;
} video_info_t;