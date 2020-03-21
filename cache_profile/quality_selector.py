class QualitSelector():
    def __init__(self, device, fps, dataset_rootdir, gop, reference_video, models, aps_class, threshold):
        self.device = device
        self.dataset_rootdir = dataset_rootdir
        self.reference_video = reference_video
        self.fps = fps
        self.gop = gop
        self.models = models
        self.aps_class = aps_class
        self.threshold = threshold

        self.latency = {}
        cache_profile = '{}_{}.profile'.format(aps_class.NAME1, threshold)
        for model in models:
            anchor_point = []
            non_anchor_frame = []

            latency_log = os.path.join(dataset_rootdir, reference_video, model.name, cache_profile, device, 'latency.txt')
            metadata_log = os.path.join(dataset_rootdir, reference_video, model.name, cache_profile, device, 'metadata_log.txt')

            with open(latency_log, 'r') as f0, open(metadata_log, 'r') as f1:
                latency_lines = f0.readlines()
                metadata_lines = f1.readlines()

                for latency_line, metadataline in zip(latency_lines, metadata_lines):
                    latency_result = latency_lines.strip().split('\t')
                    metadata_result = metadata_lines.strip().split('\t')

                    #anchor point
                    if int(metadata[2]) == 1:
                        anchor_point.append(float(latency[2]))
                    #non-anchor frame
                    elif int(metadata[2]) == 0:
                        non_anchor_frame.append(float(latency[2]))
                    else:
                        raise RuntimeError

                self.latency[model.name] = {}
                self.latency[model.name]['anchor_point'] = np.average(anchor_point)
                self.latency[model.name]['non_anchor_frame'] = np.average(non_anchor_frame)

                print('Device {}: Anchor point {:.2f}ms, Non-anchor frame {:.2f}ms'.format(device, np.average(anchor_point), np.average(non_anchor_frame)))

    def selct(self, video):
        selected_model = None

        for model in self.models:
            log = os.path.join(dataset_rootdir, reference_video, model.name, '{}_{}'.format(self.aps_class, self.threshold), 'quality.txt')
            with open(quality_log, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split('\t')
                    num_anchor_poitns = int(line[1])
                    latency = num_anchor_points * self.latency[model.name]['anchor_point'] +
                                (self.gop - num_anchor_points) * self.latency[model.name]['non_anchor_point']
                    if latency < (self.gop / self.fps):
                        selected_model = model
                    else:
                        break

        print('Video {}: {} is selected'.format(video, selected_model.name))

        return selected_model
