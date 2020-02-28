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
