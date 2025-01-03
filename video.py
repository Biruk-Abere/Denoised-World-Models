import imageio
import os
import numpy as np

# This class is designed to record videos, primarily from rendered environments like in reinforcement learning.
class VideoRecorder(object):
    
    # Initialization of the video recorder with specified dimensions, camera ID, and fps.
    # Default dimensions are 256x256 and default fps is 30.
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name  # Directory where the video will be saved
        self.height = height  # Height of the video frame
        self.width = width  # Width of the video frame
        self.camera_id = camera_id  # ID of the camera to be used for capturing (if applicable)
        self.fps = fps  # Frames per second
        self.frames = []  # List to store individual frames
    
    # Initialize or reset the frame storage and determine if recording is enabled.
    # If dir_name is None or enabled is False, recording will be disabled.
    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled
    
    # Records a frame from the given environment. The environment should have a render method.
    # The frame is captured in rgb format with specified dimensions and camera ID.
    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            self.frames.append(frame)  # Appending the captured frame to the frames list
    
    # Saves all the captured frames as a video in the specified directory with given fps.
    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)  # Full path for the video file
            imageio.mimsave(path, self.frames, fps=self.fps)  # Saving the frames as a video