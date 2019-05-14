//
// Created by hyunho on 4/15/19.
//

#include <jni.h>
#include <android/log.h>
#include <time.h>

#include <linux/limits.h>
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vpxdec.h>

#include "android_example_testlibvpx_MainActivity.h"
//#include "./vpxdec.h"
#include "./vpx_config.h"
//#include "./decode_test.h"
//#include "vpx_config.h"

#if CONFIG_LIBYUV
#include "third_party/libyuv/include/libyuv/scale.h"
#endif

#include "./args.h"
#include "./ivfdec.h"

#include "vpx_mem/vpx_mem.h"
#include "vpx/vpx_decoder.h"
#include "vpx_ports/mem_ops.h"
#include "vpx_ports/vpx_timer.h"
#include "vpx_scale/yv12config.h"
#include "vpx_dsp/psnr.h"

//#include <vpx_mem/vpx_mem.h>
//#include <vp9/vp9_iface_common.h>
//#include <vpx_scale/yv12config.h>
//#include <vpx_util/vpx_write_yuv_frame.h>
//#include <vpx_dsp/psnr.h>


///home/hyunho/MobiNAS/TestLibvpx/app/src/main/jni/libvpx/vpx_ports/mem_ops.h

#if CONFIG_VP8_DECODER || CONFIG_VP9_DECODER
#include "vpx/vp8dx.h"
#endif

#include "./md5_utils.h"

#include "./tools_common.h"
#if CONFIG_WEBM_IO
#include "./webmdec.h"
#endif
#include "./y4menc.h"

#define TAG "decode_test.c JNI"
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

struct VpxDecInputContext {
    struct VpxInputContext *vpx_input_ctx;
    struct WebmInputContext *webm_ctx;
};

#if CONFIG_VP8_DECODER
static const arg_def_t addnoise_level =
    ARG_DEF(NULL, "noise-level", 1, "Enable VP8 postproc add noise");
static const arg_def_t deblock =
    ARG_DEF(NULL, "deblock", 0, "Enable VP8 deblocking");
static const arg_def_t demacroblock_level = ARG_DEF(
    NULL, "demacroblock-level", 1, "Enable VP8 demacroblocking, w/ level");
static const arg_def_t mfqe =
    ARG_DEF(NULL, "mfqe", 0, "Enable multiframe quality enhancement");

static const arg_def_t *vp8_pp_args[] = { &addnoise_level, &deblock,
                                          &demacroblock_level, &mfqe, NULL };
#endif

#if CONFIG_LIBYUV
static INLINE int libyuv_scale(vpx_image_t *src, vpx_image_t *dst,
                               FilterModeEnum mode) {
#if CONFIG_VP9_HIGHBITDEPTH
  if (src->fmt == VPX_IMG_FMT_I42016) {
    assert(dst->fmt == VPX_IMG_FMT_I42016);
    return I420Scale_16(
        (uint16_t *)src->planes[VPX_PLANE_Y], src->stride[VPX_PLANE_Y] / 2,
        (uint16_t *)src->planes[VPX_PLANE_U], src->stride[VPX_PLANE_U] / 2,
        (uint16_t *)src->planes[VPX_PLANE_V], src->stride[VPX_PLANE_V] / 2,
        src->d_w, src->d_h, (uint16_t *)dst->planes[VPX_PLANE_Y],
        dst->stride[VPX_PLANE_Y] / 2, (uint16_t *)dst->planes[VPX_PLANE_U],
        dst->stride[VPX_PLANE_U] / 2, (uint16_t *)dst->planes[VPX_PLANE_V],
        dst->stride[VPX_PLANE_V] / 2, dst->d_w, dst->d_h, mode);
  }
#endif
  assert(src->fmt == VPX_IMG_FMT_I420);
  assert(dst->fmt == VPX_IMG_FMT_I420);
  return I420Scale(src->planes[VPX_PLANE_Y], src->stride[VPX_PLANE_Y],
                   src->planes[VPX_PLANE_U], src->stride[VPX_PLANE_U],
                   src->planes[VPX_PLANE_V], src->stride[VPX_PLANE_V], src->d_w,
                   src->d_h, dst->planes[VPX_PLANE_Y], dst->stride[VPX_PLANE_Y],
                   dst->planes[VPX_PLANE_U], dst->stride[VPX_PLANE_U],
                   dst->planes[VPX_PLANE_V], dst->stride[VPX_PLANE_V], dst->d_w,
                   dst->d_h, mode);
}
#endif

static int raw_read_frame(FILE *infile, uint8_t **buffer, size_t *bytes_read,
                          size_t *buffer_size) {
    char raw_hdr[RAW_FRAME_HDR_SZ];
    size_t frame_size = 0;

    if (fread(raw_hdr, RAW_FRAME_HDR_SZ, 1, infile) != 1) {
        if (!feof(infile)) warn("Failed to read RAW frame size\n");
    } else {
        const size_t kCorruptFrameThreshold = 256 * 1024 * 1024;
        const size_t kFrameTooSmallThreshold = 256 * 1024;
        frame_size = mem_get_le32(raw_hdr);

        if (frame_size > kCorruptFrameThreshold) {
            LOGW("Read invalid frame size (%u)\n", (unsigned int)frame_size);
            frame_size = 0;
        }

        if (frame_size < kFrameTooSmallThreshold) {
            LOGW("Warning: Read invalid frame size (%u) - not a raw file?\n",
                 (unsigned int)frame_size);
        }

        if (frame_size > *buffer_size) {
            uint8_t *new_buf = realloc(*buffer, 2 * frame_size);
            if (new_buf) {
                *buffer = new_buf;
                *buffer_size = 2 * frame_size;
            } else {
                LOGW("Failed to allocate compressed data buffer\n");
                frame_size = 0;
            }
        }
    }

    if (!feof(infile)) {
        if (fread(*buffer, 1, frame_size, infile) != frame_size) {
            LOGW("Failed to read full frame\n");
            return 1;
        }
        *bytes_read = frame_size;
    }

    return 0;
}

static int read_frame(struct VpxDecInputContext *input, uint8_t **buf,
                      size_t *bytes_in_buffer, size_t *buffer_size) {
    switch (input->vpx_input_ctx->file_type) {
#if CONFIG_WEBM_IO
        case FILE_TYPE_WEBM:
            return webm_read_frame(input->webm_ctx, buf, bytes_in_buffer);
#endif
        case FILE_TYPE_RAW:
            return raw_read_frame(input->vpx_input_ctx->file, buf, bytes_in_buffer,
                                  buffer_size);
        case FILE_TYPE_IVF:
            return ivf_read_frame(input->vpx_input_ctx->file, buf, bytes_in_buffer,
                                  buffer_size);
        default: return 1;
    }
}

static void update_image_md5(const vpx_image_t *img, const int planes[3],
                             MD5Context *md5) {
    int i, y;

    for (i = 0; i < 3; ++i) {
        const int plane = planes[i];
        const unsigned char *buf = img->planes[plane];
        const int stride = img->stride[plane];
        const int w = vpx_img_plane_width(img, plane) *
                      ((img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 2 : 1);
        const int h = vpx_img_plane_height(img, plane);

        for (y = 0; y < h; ++y) {
            MD5Update(md5, buf, w);
            buf += stride;
        }
    }
}

static void write_image_file(const vpx_image_t *img, const int planes[3],
                             FILE *file) {
    int i, y;
#if CONFIG_VP9_HIGHBITDEPTH
    const int bytes_per_sample = ((img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 2 : 1);
#else
    const int bytes_per_sample = 1;
#endif

    for (i = 0; i < 3; ++i) {
        const int plane = planes[i];
        const unsigned char *buf = img->planes[plane];
        const int stride = img->stride[plane];
        const int w = vpx_img_plane_width(img, plane);
        const int h = vpx_img_plane_height(img, plane);

        for (y = 0; y < h; ++y) {
            fwrite(buf, bytes_per_sample, w, file);
            buf += stride;
        }
    }
}

static int file_is_raw(struct VpxInputContext *input) {
    uint8_t buf[32];
    int is_raw = 0;
    vpx_codec_stream_info_t si;

    si.sz = sizeof(si);

    if (fread(buf, 1, 32, input->file) == 32) {
        int i;

        if (mem_get_le32(buf) < 256 * 1024 * 1024) {
            for (i = 0; i < get_vpx_decoder_count(); ++i) {
                const VpxInterface *const decoder = get_vpx_decoder_by_index(i);
                if (!vpx_codec_peek_stream_info(decoder->codec_interface(), buf + 4,
                                                32 - 4, &si)) {
                    is_raw = 1;
                    input->fourcc = decoder->fourcc;
                    input->width = si.w;
                    input->height = si.h;
                    input->framerate.numerator = 30;
                    input->framerate.denominator = 1;
                    break;
                }
            }
        }
    }

    rewind(input->file);
    return is_raw;
}

static void show_progress(int frame_in, int frame_out, uint64_t dx_time) {
    LOGI("%d decoded frames/%d showed frames in %" PRId64 " us (%.2f fps)\r",
         frame_in, frame_out, dx_time,
         (double)frame_out * 1000000.0 / (double)dx_time);
}

struct ExternalFrameBuffer {
    uint8_t *data;
    size_t size;
    int in_use;
};

struct ExternalFrameBufferList {
    int num_external_frame_buffers;
    struct ExternalFrameBuffer *ext_fb;
};

// Callback used by libvpx to request an external frame buffer. |cb_priv|
// Application private data passed into the set function. |min_size| is the
// minimum size in bytes needed to decode the next frame. |fb| pointer to the
// frame buffer.
static int get_vp9_frame_buffer(void *cb_priv, size_t min_size,
                                vpx_codec_frame_buffer_t *fb) {
    int i;
    struct ExternalFrameBufferList *const ext_fb_list =
            (struct ExternalFrameBufferList *)cb_priv;
    if (ext_fb_list == NULL) return -1;

    // Find a free frame buffer.
    for (i = 0; i < ext_fb_list->num_external_frame_buffers; ++i) {
        if (!ext_fb_list->ext_fb[i].in_use) break;
    }

    if (i == ext_fb_list->num_external_frame_buffers) return -1;

    if (ext_fb_list->ext_fb[i].size < min_size) {
        free(ext_fb_list->ext_fb[i].data);
        ext_fb_list->ext_fb[i].data = (uint8_t *)calloc(min_size, sizeof(uint8_t));
        if (!ext_fb_list->ext_fb[i].data) return -1;

        ext_fb_list->ext_fb[i].size = min_size;
    }

    fb->data = ext_fb_list->ext_fb[i].data;
    fb->size = ext_fb_list->ext_fb[i].size;
    ext_fb_list->ext_fb[i].in_use = 1;

    // Set the frame buffer's private data to point at the external frame buffer.
    fb->priv = &ext_fb_list->ext_fb[i];
    return 0;
}

// Callback used by libvpx when there are no references to the frame buffer.
// |cb_priv| user private data passed into the set function. |fb| pointer
// to the frame buffer.
static int release_vp9_frame_buffer(void *cb_priv,
                                    vpx_codec_frame_buffer_t *fb) {
    struct ExternalFrameBuffer *const ext_fb =
            (struct ExternalFrameBuffer *)fb->priv;
    (void)cb_priv;
    ext_fb->in_use = 0;
    return 0;
}

static void generate_filename(const char *pattern, char *out, size_t q_len,
                              unsigned int d_w, unsigned int d_h,
                              unsigned int frame_in) {
    const char *p = pattern;
    char *q = out;

    do {
        char *next_pat = strchr(p, '%');

        if (p == next_pat) {
            size_t pat_len;

            /* parse the pattern */
            q[q_len - 1] = '\0';
            switch (p[1]) {
                case 'w': snprintf(q, q_len - 1, "%d", d_w); break;
                case 'h': snprintf(q, q_len - 1, "%d", d_h); break;
                case '1': snprintf(q, q_len - 1, "%d", frame_in); break;
                case '2': snprintf(q, q_len - 1, "%02d", frame_in); break;
                case '3': snprintf(q, q_len - 1, "%03d", frame_in); break;
                case '4': snprintf(q, q_len - 1, "%04d", frame_in); break;
                case '5': snprintf(q, q_len - 1, "%05d", frame_in); break;
                case '6': snprintf(q, q_len - 1, "%06d", frame_in); break;
                case '7': snprintf(q, q_len - 1, "%07d", frame_in); break;
                case '8': snprintf(q, q_len - 1, "%08d", frame_in); break;
                case '9': snprintf(q, q_len - 1, "%09d", frame_in); break;
                default: die("Unrecognized pattern %%%c\n", p[1]); break;
            }

            pat_len = strlen(q);
            if (pat_len >= q_len - 1) die("Output filename too long.\n");
            q += pat_len;
            p += 2;
            q_len -= pat_len;
        } else {
            size_t copy_len;

            /* copy the next segment */
            if (!next_pat)
                copy_len = strlen(p);
            else
                copy_len = next_pat - p;

            if (copy_len >= q_len - 1) die("Output filename too long.\n");

            memcpy(q, p, copy_len);
            q[copy_len] = '\0';
            q += copy_len;
            p += copy_len;
            q_len -= copy_len;
        }
    } while (*p);
}

static int is_single_file(const char *outfile_pattern) {
    const char *p = outfile_pattern;

    do {
        p = strchr(p, '%');
        if (p && p[1] >= '1' && p[1] <= '9')
            return 0;  // pattern contains sequence number, so it's not unique
        if (p) p++;
    } while (p);

    return 1;
}

static void print_md5(unsigned char digest[16], const char *filename) {
    int i;

    for (i = 0; i < 16; ++i) printf("%02x", digest[i]);
    printf("  %s\n", filename);
}

static FILE *open_outfile(const char *name, const char *log_dir) {
    if (strcmp("-", name) == 0) {
        set_binary_mode(stdout);
        return stdout;
    } else {
        char file_path[PATH_MAX];
        sprintf(file_path, "%s/%s", log_dir, name);
        FILE *file = fopen(file_path, "wb");
        if (!file)
        {
            LOGE("Failed to open output file '%s'", name);
            exit(EXIT_FAILURE);
        }
        return file;
    }
}

static FILE *open_logfile(const char *name, const char *log_dir) {
    char file_path[PATH_MAX];
    sprintf(file_path, "%s/%s", log_dir, name);
    FILE *file = fopen(file_path, "w");
    if (!file) {
        LOGE("Failed to open output file '%s' with errorno '%d'", name, errno);
        exit(EXIT_FAILURE);
    }
    return file;
}

#if CONFIG_VP9_HIGHBITDEPTH
static int img_shifted_realloc_required(const vpx_image_t *img,
                                        const vpx_image_t *shifted,
                                        vpx_img_fmt_t required_fmt) {
  return img->d_w != shifted->d_w || img->d_h != shifted->d_h ||
         required_fmt != shifted->fmt;
}
#endif

int decode_test(const char *video_dir, const char *log_dir, decode_info_t video_info) {
    memset(video_info.log_dir, 0, PATH_MAX);
    sprintf(video_info.log_dir, "%s", log_dir);

    vpx_codec_ctx_t decoder;
    int i;
    int ret = EXIT_FAILURE;
    uint8_t *buf = NULL;
    size_t bytes_in_buffer = 0, buffer_size = 0;
    FILE *infile;
    int stop_after = 2, frame_in = 0, frame_out = 0, flipuv = 0, noblit = 0;
    int do_md5 = 0, progress = 0;
    int postproc = 0, summary = 0, quiet = 1; //TODO (hyunho): set stop_after by configuration
    int arg_skip = 0;
    int ec_enabled = 0;
    int keep_going = 0;
    const VpxInterface *interface = NULL;
    const VpxInterface *fourcc_interface = NULL;
    uint64_t dx_time = 0;
    struct arg arg;
    char **argv, **argi, **argj;

    int single_file;
    int use_y4m = 1;
    int opt_yv12 = 0;
    int opt_i420 = 0;
    vpx_codec_dec_cfg_t cfg = { 0, 0, 0 };
#if CONFIG_VP9_HIGHBITDEPTH
    unsigned int output_bit_depth = 0;
#endif
    int svc_decoding = 0;
    int svc_spatial_layer = 0;
#if CONFIG_VP8_DECODER
    vp8_postproc_cfg_t vp8_pp_cfg = { 0, 0, 0 };
#endif
    int frames_corrupted = 0;
    int dec_flags = 0;
    int do_scale = 0;
    vpx_image_t *scaled_img = NULL;
#if CONFIG_VP9_HIGHBITDEPTH
    vpx_image_t *img_shifted = NULL;
#endif
    int frame_avail, got_data, flush_decoder = 0;
    int num_external_frame_buffers = 0;
    struct ExternalFrameBufferList ext_fb_list = { 0, NULL };

    const char *outfile_pattern = NULL;
    char outfile_name[PATH_MAX] = { 0 };
    FILE *outfile = NULL;

    FILE *framestats_file = NULL;

    MD5Context md5_ctx;
    unsigned char md5_digest[16];

    struct VpxDecInputContext input = { NULL, NULL };
    struct VpxInputContext vpx_input_ctx;
#if CONFIG_WEBM_IO
    struct WebmInputContext webm_ctx;
    memset(&(webm_ctx), 0, sizeof(webm_ctx));
    input.webm_ctx = &webm_ctx;
#endif
    input.vpx_input_ctx = &vpx_input_ctx;

    /*******************Hyunho************************/ //TODO (hyunho): allocate memory, free memory, quality measurement, ...
    /*
    char file_path[PATH_MAX];
    YV12_BUFFER_CONFIG *original_frame = (YV12_BUFFER_CONFIG *) vpx_calloc(1, sizeof(YV12_BUFFER_CONFIG));
    //YV12_BUFFER_CONFIG *original_frame = (YV12_BUFFER_CONFIG *) vpx_calloc(1, sizeof(YV12_BUFFER_CONFIG));
    //YV12_BUFFER_CONFIG *reference_frame = (YV12_BUFFER_CONFIG *) vpx_calloc(1, sizeof(YV12_BUFFER_CONFIG));
    //YV12_BUFFER_CONFIG *compare_frame = (YV12_BUFFER_CONFIG *) vpx_calloc(1, sizeof(YV12_BUFFER_CONFIG));
    PSNR_STATS psnr_original;
    PSNR_STATS psnr_compare;

    stop_after = decode_info.stop_after;
    int save_quality = decode_info.save_quality;
    int save_serialized_frame = decode_info.save_serialized_frame;
    int save_decoded_frame = decode_info.save_decoded_frame;

    memset(file_path, 0, PATH_MAX);
    sprintf(file_path, "%s/log", decode_info.log_dir);
    FILE *log_file = fopen(file_path, "w");
    */

    interface = get_vpx_decoder_by_name("vp9");
    if (!interface)
        die("Error: Unrecognized argument (%s) to --codec\n", arg.val);
    outfile_pattern = "test.yuv";
    cfg.threads = 1;
    num_external_frame_buffers = 10;
    framestats_file = open_logfile("framestats", log_dir);
    progress = 1;
    summary = 1;
    /*******************Hyunho************************/

    if (!video_dir) {
        LOGE("No input file specified!\n");
        exit(EXIT_FAILURE);
    }

    /* Open file */
    char video_path[PATH_MAX];
    if (video_info.upsample) {sprintf(video_path, "%s/%dp_%d_bicubic.webm", video_dir, video_info.resolution, video_info.duration);}
    else {sprintf(video_path, "%s/%dp_%d.webm", video_dir, video_info.resolution, video_info.duration);}
    //sprintf(video_path, "%s/test.%s", video_dir, decode_info.format);
    infile = strcmp(video_path, "-") ? fopen(video_path, "rb") : set_binary_mode(stdin);

    if (!infile) {
        LOGE("Failed to open input file '%s'", strcmp(video_path, "-") ? video_path : "stdin");
        exit(EXIT_FAILURE);
    }

#if CONFIG_OS_SUPPORT
    /* Make sure we don't dump to the terminal, unless forced to with -o - */
    if (!outfile_pattern && isatty(fileno(stdout)) && !do_md5 && !noblit) {
        LOGE("Not dumping raw video to your terminal. Use '-o -' to "
             "override.\n");
        return EXIT_FAILURE;
    }
#endif
    input.vpx_input_ctx->file = infile;
    if (file_is_ivf(input.vpx_input_ctx))
        input.vpx_input_ctx->file_type = FILE_TYPE_IVF;
#if CONFIG_WEBM_IO
    else if (file_is_webm(input.webm_ctx, input.vpx_input_ctx))
        input.vpx_input_ctx->file_type = FILE_TYPE_WEBM;
#endif
    else if (file_is_raw(input.vpx_input_ctx))
        input.vpx_input_ctx->file_type = FILE_TYPE_RAW;
    else {
        LOGE( "Unrecognized input file type.\n");
#if !CONFIG_WEBM_IO
        fprintf(stderr, "vpxdec was built without WebM container support.\n");
#endif
        return EXIT_FAILURE;
    }

    outfile_pattern = outfile_pattern ? outfile_pattern : "-";
    single_file = is_single_file(outfile_pattern);

    if (!noblit && single_file) {
        generate_filename(outfile_pattern, outfile_name, PATH_MAX,
                          vpx_input_ctx.width, vpx_input_ctx.height, 0);
        if (do_md5)
            MD5Init(&md5_ctx);
        else
            outfile = open_outfile(outfile_name, log_dir);
    }

    if (use_y4m && !noblit) {
        if (!single_file) {
            LOGE(
                    "YUV4MPEG2 not supported with output patterns,"
                    " try --i420 or --yv12 or --rawvideo.\n");
            return EXIT_FAILURE;
        }

#if CONFIG_WEBM_IO
        if (vpx_input_ctx.file_type == FILE_TYPE_WEBM) {
            if (webm_guess_framerate(input.webm_ctx, input.vpx_input_ctx)) {
                LOGE(
                        "Failed to guess framerate -- error parsing "
                        "webm file?\n");
                return EXIT_FAILURE;
            }
        }
#endif
    }

    fourcc_interface = get_vpx_decoder_by_fourcc(vpx_input_ctx.fourcc);
    if (interface && fourcc_interface && interface != fourcc_interface)
        LOGW("Header indicates codec: %s\n", fourcc_interface->name);
    else
        interface = fourcc_interface;

    if (!interface) interface = get_vpx_decoder_by_index(0);

    dec_flags = (postproc ? VPX_CODEC_USE_POSTPROC : 0) |
                (ec_enabled ? VPX_CODEC_USE_ERROR_CONCEALMENT : 0);
    if (vpx_codec_dec_init(&decoder, interface->codec_interface(), &cfg,
                           dec_flags)) {
        LOGE( "Failed to initialize decoder: %s\n",
              vpx_codec_error(&decoder));
        goto fail2;
    }
    if (svc_decoding) {
        if (vpx_codec_control(&decoder, VP9_DECODE_SVC_SPATIAL_LAYER,
                              svc_spatial_layer)) {
            LOGE( "Failed to set spatial layer for svc decode: %s\n",
                  vpx_codec_error(&decoder));
            goto fail;
        }
    }
    //TODO (hyunho): enable multi-thread decoding
    /*
    int enable_row_mt = 1;
    if (interface->fourcc == VP9_FOURCC &&
        vpx_codec_control(&decoder, VP9D_SET_ROW_MT, enable_row_mt)) {
        fprintf(stderr, "Failed to set decoder in row multi-thread mode: %s\n",
                vpx_codec_error(&decoder));
        goto fail;
    }
    */
    if (!quiet) LOGE( "%s\n", decoder.name);

#if CONFIG_VP8_DECODER
    if (vp8_pp_cfg.post_proc_flag &&
      vpx_codec_control(&decoder, VP8_SET_POSTPROC, &vp8_pp_cfg)) {
    fprintf(stderr, "Failed to configure postproc: %s\n",
            vpx_codec_error(&decoder));
    goto fail;
  }
#endif

    if (arg_skip) LOGE( "Skipping first %d frames.\n", arg_skip);
    while (arg_skip) {
        if (read_frame(&input, &buf, &bytes_in_buffer, &buffer_size)) break;
        arg_skip--;
    }

    if (num_external_frame_buffers > 0) {
        ext_fb_list.num_external_frame_buffers = num_external_frame_buffers;
        ext_fb_list.ext_fb = (struct ExternalFrameBuffer *)calloc(
                num_external_frame_buffers, sizeof(*ext_fb_list.ext_fb));
        if (vpx_codec_set_frame_buffer_functions(&decoder, get_vp9_frame_buffer,
                                                 release_vp9_frame_buffer,
                                                 &ext_fb_list)) {
            LOGE( "Failed to configure external frame buffers: %s\n",
                  vpx_codec_error(&decoder));
            goto fail;
        }
    }

    frame_avail = 1;
    got_data = 0;

    if (framestats_file) fprintf(framestats_file, "bytes,qp\n");

    clock_t start, end;
    double cpu_time_used;
    start = clock();
    while (frame_avail || got_data) {
        vpx_codec_iter_t iter = NULL;
        vpx_image_t *img;
        struct vpx_usec_timer timer;
        int corrupted = 0;

        frame_avail = 0;
        if (!stop_after || frame_in < stop_after) {
            if (!read_frame(&input, &buf, &bytes_in_buffer, &buffer_size)) {
                frame_avail = 1;
                frame_in++;

                vpx_usec_timer_start(&timer);

                if (vpx_codec_decode(&decoder, buf, (unsigned int)bytes_in_buffer, (void *) &video_info, //TODO (hyunho): pass user_priv about log directory
                                     0)) {
                    const char *detail = vpx_codec_error_detail(&decoder);
                    LOGW("Failed to decode frame %d: %s", frame_in,
                         vpx_codec_error(&decoder));
                    if (detail) LOGW("Additional information: %s", detail);
                    corrupted = 1;
                    if (!keep_going) goto fail;
                }

                if (framestats_file) {
                    int qp;
                    if (vpx_codec_control(&decoder, VPXD_GET_LAST_QUANTIZER, &qp)) {
                        LOGW("Failed VPXD_GET_LAST_QUANTIZER: %s",
                             vpx_codec_error(&decoder));
                        if (!keep_going) goto fail;
                    }
                    fprintf(framestats_file, "%d,%d\n", (int)bytes_in_buffer, qp);
                }

                vpx_usec_timer_mark(&timer);
                dx_time += vpx_usec_timer_elapsed(&timer);
            } else {
                flush_decoder = 1;
            }
        } else {
            flush_decoder = 1;
        }

        vpx_usec_timer_start(&timer);

        if (flush_decoder) {
            // Flush the decoder in frame parallel decode.
            if (vpx_codec_decode(&decoder, NULL, 0, NULL, 0)) {
                LOGW("Failed to flush decoder: %s", vpx_codec_error(&decoder));
                corrupted = 1;
                if (!keep_going) goto fail;
            }
        }

        got_data = 0;
        if ((img = vpx_codec_get_frame(&decoder, &iter))) {
            ++frame_out;
            got_data = 1;
        }

        vpx_usec_timer_mark(&timer);
        dx_time += (unsigned int)vpx_usec_timer_elapsed(&timer);

        if (!corrupted &&
            vpx_codec_control(&decoder, VP8D_GET_FRAME_CORRUPTED, &corrupted)) {
            LOGW("Failed VP8_GET_FRAME_CORRUPTED: %s", vpx_codec_error(&decoder));
            if (!keep_going) goto fail;
        }
        frames_corrupted += corrupted;

        if (progress) show_progress(frame_in, frame_out, dx_time);

        /*******************Hyunho************************/
        /*
        if (image2yuvconfig(img, original_frame))
        {
            LOGE("convert image to yuv fail");
            goto DEBUG_EXIT;
        }

        if (save_serialized_frame) {
            memset(file_path, 0, sizeof(char) * PATH_MAX);
            if (decode_info.upsample == 1) sprintf(file_path, "%s/%dp_%d_upsample", decode_info.log_dir, decode_info.resolution, frame_out);
            else sprintf(file_path, "%s/%dp_%d", decode_info.log_dir, decode_info.resolution, frame_out);

            if (vpx_serialize_save(file_path, original_frame))
            {
                LOGE("serialization fail");
            };
        }

        if (save_decoded_frame) {
            memset(file_path, 0, sizeof(char) * PATH_MAX);
            if (decode_info.upsample == 1) sprintf(file_path, "%s/%dp_%d_upsample_y", decode_info.log_dir, decode_info.resolution, frame_out);
            else sprintf(file_path, "%s/%dp_%dy", decode_info.log_dir, decode_info.resolution, frame_out);

            if (vpx_write_y_frame(file_path, original_frame)) //TODO (hyunho): check whether using frame_to_show is valid or not
            {
                LOGE("vpx_write_y_frame fail");
            }
        }

        if (save_quality) {
            memset(file_path, 0, sizeof(char) * PATH_MAX);
            sprintf(file_path, "%s/%dp_%d_y", decode_info.log_dir, decode_info.resolution * decode_info.scale, frame_out);
            if(vpx_deserialize_load(compare_frame, file_path, original_frame->y_crop_width * decode_info.scale, original_frame->y_crop_height * decode_info.scale,
                    original_frame->subsampling_x, original_frame->subsampling_y, 0)) //TODO (hyunho): check byte alignment value
            {
                LOGE("deserailize fail");
                goto DEBUG_EXIT;
            }

            memset(file_path, 0, sizeof(char) * PATH_MAX);
            sprintf(file_path, "%s/%dp_%d", decode_info.log_dir, decode_info.resolution * decode_info.scale, frame_out);
            if(vpx_deserialize_load(reference_frame, file_path, original_frame->y_crop_width * decode_info.scale, original_frame->y_crop_height * decode_info.scale,
                                    original_frame->subsampling_x, original_frame->subsampling_y, 0)) //TODO (hyunho): check byte alignment value
            {
                LOGE("deserailize fail");
                goto DEBUG_EXIT;
            }


            vpx_calc_psnr(original_frame, reference_frame, &psnr_original);
            vpx_calc_psnr(compare_frame, reference_frame, &psnr_compare);

            LOGI("%d frame: original %.2fdB, compare %.2fdB", frame_out, psnr_original.psnr[0], psnr_compare.psnr[0]);
            fprintf(log_file, "%.2f\t%.2f\t%.2f\n", psnr_original.psnr[0], psnr_compare.psnr[0], (double)frame_out * 1000000.0 / (double)dx_time);
        }

        DEBUG_EXIT:
        */
        /*******************Hyunho************************/

        if (!noblit && img) {
            const int PLANES_YUV[] = { VPX_PLANE_Y, VPX_PLANE_U, VPX_PLANE_V };
            const int PLANES_YVU[] = { VPX_PLANE_Y, VPX_PLANE_V, VPX_PLANE_U };
            const int *planes = flipuv ? PLANES_YVU : PLANES_YUV;

            if (do_scale) {
                if (frame_out == 1) {
                    // If the output frames are to be scaled to a fixed display size then
                    // use the width and height specified in the container. If either of
                    // these is set to 0, use the display size set in the first frame
                    // header. If that is unavailable, use the raw decoded size of the
                    // first decoded frame.
                    int render_width = vpx_input_ctx.width;
                    int render_height = vpx_input_ctx.height;
                    if (!render_width || !render_height) {
                        int render_size[2];
                        if (vpx_codec_control(&decoder, VP9D_GET_DISPLAY_SIZE,
                                              render_size)) {
                            // As last resort use size of first frame as display size.
                            render_width = img->d_w;
                            render_height = img->d_h;
                        } else {
                            render_width = render_size[0];
                            render_height = render_size[1];
                        }
                    }
                    scaled_img =
                            vpx_img_alloc(NULL, img->fmt, render_width, render_height, 16);
                    scaled_img->bit_depth = img->bit_depth;
                }

                if (img->d_w != scaled_img->d_w || img->d_h != scaled_img->d_h) {
#if CONFIG_LIBYUV
                    libyuv_scale(img, scaled_img, kFilterBox);
          img = scaled_img;
#else
                    LOGE(
                            "Failed  to scale output frame: %s.\n"
                            "Scaling is disabled in this configuration. "
                            "To enable scaling, configure with --enable-libyuv\n",
                            vpx_codec_error(&decoder));
                    goto fail;
#endif
                }
            }
#if CONFIG_VP9_HIGHBITDEPTH
            // Default to codec bit depth if output bit depth not set
      if (!output_bit_depth && single_file && !do_md5) {
        output_bit_depth = img->bit_depth;
      }
      // Shift up or down if necessary
      if (output_bit_depth != 0 && output_bit_depth != img->bit_depth) {
        const vpx_img_fmt_t shifted_fmt =
            output_bit_depth == 8
                ? img->fmt ^ (img->fmt & VPX_IMG_FMT_HIGHBITDEPTH)
                : img->fmt | VPX_IMG_FMT_HIGHBITDEPTH;
        if (img_shifted &&
            img_shifted_realloc_required(img, img_shifted, shifted_fmt)) {
          vpx_img_free(img_shifted);
          img_shifted = NULL;
        }
        if (!img_shifted) {
          img_shifted =
              vpx_img_alloc(NULL, shifted_fmt, img->d_w, img->d_h, 16);
          img_shifted->bit_depth = output_bit_depth;
        }
        if (output_bit_depth > img->bit_depth) {
          vpx_img_upshift(img_shifted, img, output_bit_depth - img->bit_depth);
        } else {
          vpx_img_downshift(img_shifted, img,
                            img->bit_depth - output_bit_depth);
        }
        img = img_shifted;
      }
#endif

            if (single_file) {
                if (use_y4m) {
                    char buf[Y4M_BUFFER_SIZE] = { 0 };
                    size_t len = 0;
                    if (img->fmt == VPX_IMG_FMT_I440 || img->fmt == VPX_IMG_FMT_I44016) {
                        LOGE( "Cannot produce y4m output for 440 sampling.\n");
                        goto fail;
                    }
                    if (frame_out == 1) {
                        // Y4M file header
                        len = y4m_write_file_header(
                                buf, sizeof(buf), vpx_input_ctx.width, vpx_input_ctx.height,
                                &vpx_input_ctx.framerate, img->fmt, img->bit_depth);
                        if (do_md5) {
                            MD5Update(&md5_ctx, (md5byte *)buf, (unsigned int)len);
                        } else {
                            fputs(buf, outfile);
                        }
                    }

                    // Y4M frame header
                    len = y4m_write_frame_header(buf, sizeof(buf));
                    if (do_md5) {
                        MD5Update(&md5_ctx, (md5byte *)buf, (unsigned int)len);
                    } else {
                        fputs(buf, outfile);
                    }
                } else {
                    if (frame_out == 1) {
                        // Check if --yv12 or --i420 options are consistent with the
                        // bit-stream decoded
                        if (opt_i420) {
                            if (img->fmt != VPX_IMG_FMT_I420 &&
                                img->fmt != VPX_IMG_FMT_I42016) {
                                LOGE( "Cannot produce i420 output for bit-stream.\n");
                                goto fail;
                            }
                        }
                        if (opt_yv12) {
                            if ((img->fmt != VPX_IMG_FMT_I420 &&
                                 img->fmt != VPX_IMG_FMT_YV12) ||
                                img->bit_depth != 8) {
                                LOGE( "Cannot produce yv12 output for bit-stream.\n");
                                goto fail;
                            }
                        }
                    }
                }

                if (do_md5) {
                    update_image_md5(img, planes, &md5_ctx);
                } else {
                    if (!corrupted) write_image_file(img, planes, outfile);
                }
            } else {
                generate_filename(outfile_pattern, outfile_name, PATH_MAX, img->d_w,
                                  img->d_h, frame_in);
                if (do_md5) {
                    MD5Init(&md5_ctx);
                    update_image_md5(img, planes, &md5_ctx);
                    MD5Final(md5_digest, &md5_ctx);
                    print_md5(md5_digest, outfile_name);
                } else {
                    outfile = open_outfile(outfile_name, log_dir);
                    write_image_file(img, planes, outfile);
                    fclose(outfile);
                }
            }
        }
    }
    end = clock();
    LOGD("elapsed_time: %f", ((double) (end - start)) / CLOCKS_PER_SEC);

    if (summary || progress) {
        show_progress(frame_in, frame_out, dx_time);
    }

    if (frames_corrupted) {
        LOGE( "WARNING: %d frames corrupted.\n", frames_corrupted);
    } else {
        ret = EXIT_SUCCESS;
    }

    fail:

    if (vpx_codec_destroy(&decoder)) {
        LOGE( "Failed to destroy decoder: %s\n",
              vpx_codec_error(&decoder));
    }

    fail2:

    if (!noblit && single_file) {
        if (do_md5) {
            MD5Final(md5_digest, &md5_ctx);
            print_md5(md5_digest, outfile_name);
        } else {
            fclose(outfile);
        }
    }

#if CONFIG_WEBM_IO
    if (input.vpx_input_ctx->file_type == FILE_TYPE_WEBM)
        webm_free(input.webm_ctx);
#endif

    if (input.vpx_input_ctx->file_type != FILE_TYPE_WEBM) free(buf);

    if (scaled_img) vpx_img_free(scaled_img);
#if CONFIG_VP9_HIGHBITDEPTH
    if (img_shifted) vpx_img_free(img_shifted);
#endif

    for (i = 0; i < ext_fb_list.num_external_frame_buffers; ++i) {
        free(ext_fb_list.ext_fb[i].data);
    }
    free(ext_fb_list.ext_fb);

    fclose(infile);
    if (framestats_file) fclose(framestats_file);

    free(argv);

    LOGI("main_loop success");

    /*******************Hyunho************************/
    /*
    free(original_frame);
    free(reference_frame);
    free(compare_frame);
    */
    /*******************Hyunho************************/

    return ret;
}