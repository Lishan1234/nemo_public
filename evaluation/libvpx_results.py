import os

def libvpx_quality(log_dir):
        quality_log_file = os.path.join(log_dir, 'quality.txt')
        quality = []

        with open(quality_log_file, 'r') as f:
            quality_lines = f.readlines()

            for quality_line in quality_lines:
                quality_line = quality_line.strip().split('\t')
                quality.append(float(quality_line[1]))

        return quality

def libvpx_latency(log_dir):
        latency_log_file = os.path.join(log_dir, 'latency.txt')
        latency = []

        with open(latency_log_file, 'r') as f:
            latency_lines = f.readlines()

            for latency_line in latency_lines:
                latency_line = latency_line.strip().split('\t')
                latency.append(float(latency_line[2]))

        return latency
