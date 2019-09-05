//
// Created by hyunho on 6/13/19.
//

#ifndef LIBVPX_WRAPPER_DECODE_TEST_H
#define LIBVPX_WRAPPER_DECODE_TEST_H

#include <vpx/vpx_decoder.h>
#include <vpx/vpx_mobinas.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int decode_test(mobinas_cfg_t *mobinas_cfg);

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //LIBVPX_WRAPPER_DECODE_TEST_H
