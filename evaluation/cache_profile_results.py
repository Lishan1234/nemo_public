import os
import numpy as np

def chunk_quality(log_dir):
    quality_log_file = os.path.join(log_dir, 'quality.txt')
    quality_cache = []
    quality_dnn = None

    with open(quality_log_file, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality_cache.append(float(quality_line[1]))
            quality_dnn = float(quality_line[2])

    return quality_cache, quality_dnn

def chunk_estimated_quality(log_dir):
    quality_log_file = os.path.join(log_dir, 'quality.txt')
    quality_estimated = []

    with open(quality_log_file, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality_estimated.append(float(quality_line[4]))

    return quality_estimated

def chunk_error_frames(log_dir):
    error_log_file = os.path.join(log_dir, 'error.txt')
    error_cache = []

    with open(error_log_file, 'r') as f:
        error_lines = f.readlines()

        for error_line in error_lines:
            error_line = error_line.strip().split('\t')
            error_cache.append(int(error_line[2]))

    return error_cache

def chunk_anchor_points(log_dir):
    error_log_file = os.path.join(log_dir, 'error.txt')
    error_cache = []

    with open(error_log_file, 'r') as f:
        error_lines = f.readlines()
        error_line = error_lines[-1].strip().split('\t')
        num_anchor_points = error_line[0]

    return num_anchor_points

def chunk_anchor_point_index(log_dir):
    metadata_log_file = os.path.join(log_dir, 'metadata.txt')
    anchor_point_index = []

    with open(metadata_log_file, 'r') as f:
        metadata_lines = f.readlines()
        for idx, metadata_line in enumerate(metadata_lines):
            metadata_line = metadata_line.strip().split('\t')
            if int(metadata_line[2]) == 1:
                anchor_point_index.append(idx)

    return anchor_point_index

def chunk_out_degree(log_dir, anchor_point_index):
    metadata_log_file = os.path.join(log_dir, 'out_degree_per_frame.txt')
    out_degree = []

    with open(metadata_log_file, 'r') as f:
        metadata_lines = f.readlines()
        for idx, metadata_line in enumerate(metadata_lines):
            metadata_line = metadata_line.strip().split('\t')
            if anchor_point_index is None:
                out_degree.append(int(metadata_line[2]))
            else:
                if idx in anchor_point_index:
                    out_degree.append(int(metadata_line[2]))

    return out_degree

def quality(log_dir):
    quality_log_file = os.path.join(log_dir, 'quality.txt')
    quality_cache = []
    quality_dnn = []
    quality_bilinear = []

    with open(quality_log_file, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality_cache.append(float(quality_line[2]))
            quality_dnn.append(float(quality_line[3]))
            quality_bilinear.append(float(quality_line[4]))

    result_cache = {}
    result['avg_quality'] = np.round(np.average(quality_cache), 4)
    result['std_quality'] = np.round(np.std(quality_cache), 4)
    result_dnn = {}
    result['avg_quality'] = np.round(np.average(quality_dnn), 4)
    result['std_quality'] = np.round(np.std(quality_dnn), 4)
    result_bilinear = {}
    result['avg_quality'] = np.round(np.average(quality_bilinear), 4)
    result['std_quality'] = np.round(np.std(quality_bilinear), 4)

    return result_cache, result_dnn, result_bilinear

def quality_gain(log_dir):
    quality_log_file = os.path.join(log_dir, 'quality.txt')
    quality_gain_cache = []
    quality_gain_dnn = []

    with open(quality_log_file, 'r') as f:
        quality_lines = f.readlines()

        for quality_line in quality_lines:
            quality_line = quality_line.strip().split('\t')
            quality_cache = float(quality_line[2])
            quality_dnn = float(quality_line[3])
            quality_bilinear = float(quality_line[4])
            quality_gain_cache.append(quality_cache - quality_bilinear)
            quality_gain_dnn.append(quality_dnn - quality_bilinear)

    result_cache = {}
    result['avg_quality'] = np.round(np.average(quality_cache_gain), 4)
    result['std_quality'] = np.round(np.std(quality_cache_gain), 4)
    result_dnn = {}
    result['avg_quality'] = np.round(np.average(quality_dnn_gain), 4)
    result['std_quality'] = np.round(np.std(quality_dnn_gain), 4)

    return result_cache, result_dnn, result_bilinear

def norm_mac(log_dir):
    mac_log_file = os.path.join(log_dir, 'mac.txt')
    mac_cache = []
    mac_dnn = []

    with open(mac_log_file, 'r') as f:
        mac_lines = f.readlines()

        for mac_line in mac_lines:
            mac_line = mac_line.strip().split('\t')
            mac_cache.append(float(mac_line[2]))
            mac_dnn.append(float(mac_line[3]))

    result_cache = {}
    result_cache['avg_mac'] = np.round(np.average(np.divide(np.asarray(mac_cache), np.asarray(mac_dnn))), 4)
    result_cache['std_mac'] = np.round(np.std(np.divide(np.asarray(mac_cache), np.asarray(mac_dnn))), 4)

    result_dnn = {}
    result_dnn['avg_mac'] = np.round(np.average(np.divide(np.asarray(mac_dnn), np.asarray(mac_dnn))), 4)
    result_dnn['std_mac'] = np.round(np.std(np.divide(np.asarray(mac_dnn), np.asarray(mac_dnn))), 4)

    return result_cache, result_dnn

def mac(log_dir):
    mac_log_file = os.path.join(log_dir, 'mac.txt')
    mac_cache = []
    mac_dnn = []

    with open(mac_log_file, 'r') as f:
        mac_lines = f.readlines()

        for mac_line in mac_lines:
            mac_line = mac_line.strip().split('\t')
            mac_cache.append(float(mac_line[2]))
            mac_dnn.append(float(mac_line[3]))

    result_cache = {}
    result_cache['avg_mac'] = np.round(np.average(mac_cache), 4)
    result_cache['std_mac'] = np.round(np.std(mac_cache), 4)

    result_dnn = {}
    result_dnn['avg_mac'] = np.round(np.average(mac_dnn), 4)
    result_dnn['std_mac'] = np.round(np.std(mac_dnn), 4)

    return result_cache, result_dnn
