#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from pathlib import Path
import numpy as np

meta_keys = [
    'videoStart', 'videoEnd', 'totalFrames', 
    'frameRate', 'skipped', 'maxSpan'
]

joint_keys = [
    'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',
    'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 
    'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle',
    'head', 'torso'
]

joint_indeces = [
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
]


# In[2]:


class LMAVideoSegment(object):
    
    def __init__(self, joint_data, meta_data, segmentation_thresholds=None, joint_keys=None):
        self.joint_data = joint_data
        self.meta_data = meta_data
        self.segmentation_thresholds = segmentation_thresholds
        self.joint_keys = joint_keys
        
    def __str__(self):
        m = self.meta_data
        tmp = '{0:15}|{1:15}|{2:15}|{3:15}'
        longstring = tmp.format('Frames', 'FrameRate', 'Number of Keys', 'Keys') + '\n'
        longstring += tmp.format(m['totalFrames'], m['frameRate'], 'nan', 'nan')
        return longstring
        
class LMAJSONUtil(object):
    
    wheel = ['-', '-', '-', '\\', '\\', '\\', '|', '|', '|', '/', '/', '/']
    
    def __init__(self, joint_keys=None, joint_indeces=None, meta_keys=None, progress_callback=None):
        self.joint_keys = joint_keys
        self.joint_indeces = joint_indeces
        self.meta_keys = meta_keys
        self.progress_callback = progress_callback
    
    def __glob_json(self, relative_path, process_callback):
        p = Path(relative_path)
        fcounter = 0
        for file in p.glob('*.json'):
            w = LMAJSONUtil.wheel[int(fcounter % len(LMAJSONUtil.wheel))]
            print('\r[*] Processing: {0}'.format(w), end='', flush=True)
            fcounter += 1
            yield process_callback(relative_path + file.name)
        
        print('\r[*]  Processed {0} files'.format(fcounter))
        
    def __open_json(self, file):
        with open(file) as file:
            return json.load(file)
    
    def __coordgen(self, frame_array):
        for fidx, frame in enumerate(frame_array):
            if frame['interpolation']:
                k = list(frame.keys())
                ex = ['head', 'torso']
                if not all(el in k for el in ex):
                    print('Skipped frame {0}'.format(fidx))
                    continue
                
            yield ('head', frame['head'])
            yield ('torso', frame['torso'])
            
            coords = frame['data']
            for idx, key in zip(self.joint_indeces, self.joint_keys):
                yield (key, coords[idx])
                
    def __on_process_video_json(self, file):
        json = self.__open_json(file)
        
        p = self.joint_keys
        q = [[] for n in range(len(p))]
        d = dict(zip(p, q))
        frames = json['frames']
                
        for c in self.__coordgen(frames):
            d[c[0]].append(c[1])
        
        return {
            'meta': {k:json[k] for k in self.meta_keys},
            'data': {k:np.asarray(v, dtype=np.float) for k, v in d.items()}
        }
        
    def __load_threshold_json(self, file):
        
        if file is None:
            return {}
        
        return self.__open_json(file)
    
    def get_video_segments(self, path_video, path_thresholds):
        segments = []
        t = self.__load_threshold_json(path_thresholds)
        for p in self.__glob_json(path_video, self.__on_process_video_json):
            seg = LMAVideoSegment(
                joint_data=p['data'],
                meta_data=p['meta'],
                segmentation_thresholds=t,
                joint_keys=self.joint_keys
            )
            segments.append(seg)
        
        return segments
    

