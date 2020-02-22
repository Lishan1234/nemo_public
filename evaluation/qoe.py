import os
import numpy as np

def bilinear_quality(dataset_dir, lr_video_name):
        log_dir = os.path.join(dataset_dir, 'log', lr_video_name, model.name)
        quality_log_file = os.path.join(log_dir, 'quality.txt')
        psnr_values = []

        with open(quality_log_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                psnr_value = float(line.strip().split('\t')[1])

        result = {}
        result['avg_quality'] = np.average(psnr_values)
        result['std_quality'] = np.std(psnr_values)

        return result
