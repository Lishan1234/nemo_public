import os
import numpy as np

def mac_and_quality_gain(log_dir):
        quality_log_file = os.path.join(log_dir, 'quality.txt')
        mac_log_file = os.path.join(log_dir, 'mac.txt')
        quality_gain_cache = []
        quality_gain_dnn = []
        mac_cache = []
        mac_dnn = []

        with open(quality_log_file, 'r') as qf, open(mac_log_file, 'r') as mf:
            quality_lines = qf.readlines()
            mac_lines = mf.readlines()

            for quality_line, mac_line in zip(quality_lines, mac_lines):
                quality_line = quality_line.strip().split('\t')
                quality_cache = quality_line[2]
                quality_dnn = quality_line[3]
                quality_bilinear = quality_line[4]
                quality_gain_cache.append(float(quality_cache) - float(quality_bilinear))
                quality_gain_dnn.append(float(quality_dnn) - float(quality_bilinear))
                mac_line = mac_line.strip().split('\t')
                mac_cache.append(float(mac_line[2]))
                mac_dnn.append(float(mac_line[3]))

        result = {}
        result['avg_quality'] = {}
        result['std_quality'] = {}
        result['avg_mac'] = {}
        result['std_mac'] = {}
        result['avg_norm_mac'] = {}
        result['std_norm_mac'] = {}
        result['avg_quality']['cache'] = np.average(quality_gain_cache)
        result['avg_quality']['dnn'] = np.average(quality_gain_dnn)
        result['std_quality']['cache'] = np.std(quality_gain_cache)
        result['std_quality']['dnn'] = np.std(quality_gain_dnn)
        result['avg_mac']['cache'] = np.average(mac_cache)
        result['avg_mac']['dnn'] = np.average(mac_dnn)
        result['std_mac']['cache'] = np.std(mac_cache)
        result['std_mac']['dnn'] = np.std(mac_dnn)
        result['avg_norm_mac']['cache'] = np.average(mac_cache) / np.average(mac_dnn)
        result['avg_norm_mac']['dnn'] = np.average(mac_dnn) / np.average(mac_dnn)
        result['std_norm_mac']['cache'] = np.std(np.divide(np.asarray(mac_cache), np.asarray(mac_dnn)))
        result['std_norm_mac']['dnn'] = np.std(np.divide(np.asarray(mac_dnn), np.asarray(mac_dnn)))

        return result
