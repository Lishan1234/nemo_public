import os
import sys

def libvpx_quality(log_dir):
        quality_log_file = os.path.join(log_dir, 'quality.txt')
        quality = []

        with open(quality_log_file, 'r') as f:
            quality_lines = f.readlines()

            for quality_line in quality_lines:
                quality_line = quality_line.strip().split('\t')
                quality.append(float(quality_line[1]))

        return quality

def libvpx_anchor_points(log_dir):
        quality_log_file = os.path.join(log_dir, 'quality.txt')
        anchor_points = []

        with open(quality_log_file, 'r') as f:
            quality_lines = f.readlines()

            for quality_line in quality_lines:
                quality_line = quality_line.strip().split('\t')
                anchor_points.append(float(quality_line[1]))

        return anchor_points

def libvpx_latency(log_dir):
        latency_log_file = os.path.join(log_dir, 'latency.txt')
        latency = []

        with open(latency_log_file, 'r') as f:
            latency_lines = f.readlines()

            for latency_line in latency_lines:
                latency_line = latency_line.strip().split('\t')
                latency.append(float(latency_line[2]))

        return latency

def libvpx_mac(log_dir):
        mac_log_file = os.path.join(log_dir, 'mac.txt')
        cache_mac = []
        dnn_mac = []

        with open(mac_log_file, 'r') as f:
            mac_lines = f.readlines()

            for mac_line in mac_lines:
                mac_line = mac_line.strip().split('\t')
                cache_mac.append(float(mac_line[2]))
                dnn_mac.append(float(mac_line[3]))

        return cache_mac, dnn_mac

def libvpx_breakdown_latency(log_dir):
        latency_log_file = os.path.join(log_dir, 'latency_thread04.txt')
        decode = []
        bilinear_interpolation = []
        motion_compensation = []

        with open(latency_log_file, 'r') as f:
            latency_lines = f.readlines()

            for latency_line in latency_lines:
                latency_line = latency_line.strip().split('\t')
                decode_latency = 0
                decode_latency += float(latency_line[2])
                decode_latency += float(latency_line[3])
                decode_latency += float(latency_line[4])
                decode.append(decode_latency)

                bilinear_interpolation_latency = 0
                bilinear_interpolation_latency += float(latency_line[5])
                bilinear_interpolation_latency += float(latency_line[7])
                bilinear_interpolation.append(bilinear_interpolation_latency)

                motion_compensation.append(float(latency_line[6]))

        return decode, bilinear_interpolation, motion_compensation

def libvpx_num_frames(log_dir):
        log_file = os.path.join(log_dir, 'metadata.txt')

        with open(log_file, 'r') as f:
            lines = f.readlines()

        return len(lines)

def libvpx_power(log_file):
        time = []
        current = []
        power = []

        with open(log_file, 'r') as f:
            lines = f.readlines()

            for idx, line in enumerate(lines):
                if idx == 0 :
                   continue
                else:
                    results = line.strip().split(',')
                    time.append(float(results[0]))
                    current.append(float(results[1]))
                    power.append(float(results[2]))

        return time[-1] - time[0], current, power

def libvpx_temperature(log_file):
        time = []
        current = []
        temperature = []

        with open(log_file, 'r') as f:
            lines = f.readlines()

            for idx, line in enumerate(lines):
                if idx == 0 :
                   continue
                else:
                    results = line.strip().split(',')
                    time.append(float(results[0]))
                    temperature.append(float(results[4]))

        return time, temperature
