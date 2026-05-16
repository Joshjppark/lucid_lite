import numpy as np
from pathlib import Path
import sleap_io as sio


from geometry import *



class LucidLabels:
    
    def __init__(self, filename):
        self.filename = filename
        self.labels = sio.load_file(filename)
        self._frameGroups = None
        self.camera_mapping = [Path(v.filename).name for v in self.labels.videos]

    @property
    def frameGroups(self):
        if self._frameGroups is None:
            frame_groups_map = {}
            for lf in self.labels.labeled_frames:
                idx = lf.frame_idx

                if idx not in frame_groups_map:
                    frame_groups_map[idx] = FrameInstances(
                        idx,
                        self.camera_mapping,
                        self.labels.sessions[0].camera_group
                        )
                frame_instance = frame_groups_map.get(idx)

                camera_name = Path(lf.video.filename).name
                camera_pos = self.camera_mapping.index(camera_name)

                for instance in lf.instances:
                    frame_instance.points.append(
                        (instance.points['xy'], instance.score, camera_name)
                    )
                frame_instance.shape[camera_pos] = len(lf.instances)

            self._frameGroups = frame_groups_map

        return self._frameGroups



class FrameInstances:

    def __init__(self, frameIdx, camera_mapping, calibration):
        self.frameIdx = frameIdx
        self.n_cameras = len(camera_mapping)
        self.camera_mapping = camera_mapping
        self.points =   [] # each entry is (Instance, score, camera)
        self.shape =    [np.nan] * self.n_cameras

        self.calibration = calibration


    def __repr__(self):
        return f'Frame idx: {self.frameIdx}\n' \
               f'Shape: {self.shape}\n' \
               f'Points: {self.points}'



class IdentitySolver:

    def __init__(self, labels):
        self.labels = labels
        self.epipolar_threshold = 10

    
    def get_highest_score(self, idx: int):
        frame = self.labels.frameGroups[idx]
        frame.points = sorted(frame.points, key=lambda x: x[1], reverse=True)

        anchors = []

        for instance_tup in frame.points:
            
            # first instance is always anchor because it has highest score
            if anchors is []:
                anchors = instance_tup
                continue

            is_new_anchor = True

            for anchor in anchors:
                # now, we want to find the next anchor. this means one of two things
                # 1. instance is in the same view as the anchor
                # 2. instance has high epipolar distance from other anchors (not same identity)
                if anchor[2] == instance_tup[2]:
                    continue

                dist = calculate_epipolar_distance(anchor[0], instance_tup[0], cameras)

                print(f'dist is {dist}')

                if dist < self.epipolar_threshold:
                    is_new_anchor = False
                    break

            if is_new_anchor:
                anchors.append(instance_tup)

            
        return anchors





def main():

    print('hello world')

if __name__ == '__main__':
    main()
    