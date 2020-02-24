def quality(log_dir):
        quality_log_file = os.path.join(log_dir, 'quality.txt')
        quality = []

        with open(quality_log_file, 'r') as f:
            quality_lines = f.readlines()

            for quality_line and quality_lines:
                quality_line = quality_line.strip().split('\t')
                quality.append(float(quality_line[2]))

        result = {}
        result['avg_quality'] = np.average(quality)

        return result
