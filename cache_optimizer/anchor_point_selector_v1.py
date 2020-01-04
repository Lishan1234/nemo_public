class APS_v1:
    def __init__(self, cra):
        self.cra = cra

    #TODO: measure time for each process
    def run(self, chunk_idx):
        #setup cra
        self.cra.prepare(chunk_idx)

        #profile anchor points
        anchor_points = self.cra.profile_anchor_points(chunk_idx)

        #select anchor points
        cache_profiles = []
        cache_profile = None
        while len(anchor_points) > 0:
            cache_profile = self._select_anchor_point(cache_profile, anchor_points)
            cache_profiles.append(cache_profile)

        #select/save a cache profile
        self.cra.select_cache_profile(chunk_idx, cache_profiles, quality_diff)

    def _select_anchor_point(self, cache_profile, anchor_points):
        max_avg_quality = 0
        target_anchor_point = None

        for anchor_point in anchor_points:
            quality = self._estimate_quality(cache_profile, anchor_point)
            avg_quality = np.average(quality)

            if avg_quality > max_avg_quality:
                target_anchor_point = anchor_point
                max_avg_quality = avg_quality

        new_cache_profile = CacheProfile(cache_profile, self.name)
        new_cache_profile.add_anchor_point(target_anchor_point)

        return new_cache_profile

    def _estimate_quality(self, cache_profile, anchor_point):
        if cache_profile is not None:
            return np.maximum(cache_profile.quality, anchor_point.quality)
        else:
            return anchor_point.quality

    @property
    def name(self):
        return 'aps_v1'
