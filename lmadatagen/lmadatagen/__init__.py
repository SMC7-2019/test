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

from .LMAJSONUtil import LMAJSONUtil
from .LMAApproximator import LMARunner
