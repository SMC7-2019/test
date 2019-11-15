#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path
import numpy as np

class LMAVideoSegment(object):
    
    def __init__(self, joint_data=None, meta_data=None, joint_keys=None):
        self.joint_data = joint_data
        self.meta_data = meta_data
        self.joint_keys = joint_keys
        
    def recon(self, _dict):
        self.joint_data = {
            k:np.asarray(v, dtype=np.float) for k, v in _dict['joint_data'].items()
        }
        self.meta_data = _dict['meta_data']
        self.joint_keys = _dict['joint_keys']
    
class LMAJSONUtil(object):
    
    wheel = ['-', '-', '-', '\\', '\\', '\\', '|', '|', '|', '/', '/', '/']
    
    def __init__(self, joint_keys=None, joint_indeces=None, meta_keys=None, local_save=None):
        self.joint_keys = joint_keys
        self.joint_indeces = joint_indeces
        self.meta_keys = meta_keys
        self.local_save_file = 'video_segments.json'
        
        self.__segments = []
        
    def __glob_json(self, relative_path, process_callback):
        p = Path(relative_path)
        fcounter = 0
        for file in p.glob('*.json'):
            w = LMAJSONUtil.wheel[int(fcounter % len(LMAJSONUtil.wheel))]
            print('\r[*] Processing: {0}'.format(w), end='', flush=True)
            fcounter += 1
            yield process_callback(relative_path + file.name)
        
        print('\r[*]  Processed {0} files'.format(fcounter), end='', flush=True)
        
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
                
    def __on_process_motion_json(self, file):
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
    
    def __save_motion_json(self):
        json_string = json.dumps(
            [ob.__dict__ for ob in self.__segments], 
            default=lambda _obj: _obj.tolist()
        )
        
        with open(self.local_save_file, 'w', encoding='utf8') as f:
            f.write(str(json_string))
        
    def __process_motion_json(self, path_motion, save_to_disc=False):
        for p in self.__glob_json(path_motion, self.__on_process_motion_json):
            seg = LMAVideoSegment(
                joint_data=p['data'],
                meta_data=p['meta'],
                joint_keys=self.joint_keys
            )
            self.__segments.append(seg)        
        
        if save_to_disc:
            self.__save_motion_json()
    
    def __load_local(self):
        with open(self.local_save_file, 'r') as f:
            data_loaded = json.loads(f.read())
            for data in data_loaded:
                _obj = LMAVideoSegment()
                _obj.recon(data)
                self.__segments.append(_obj)
    
    def __load_motion_json(self):
        try:
            self.__load_local()
        except FileNotFoundError:
            print('No saved file on disc')
            
    def get_video_segments(self, path_motion=None, save_disc=False, ret_disc=True):
        if not ret_disc:
            self.__process_motion_json(path_motion, save_disc)
        else:
            self.__load_motion_json()
        
        return self.__segments 
