import distutils.spawn
import distutils.version
import os
import os.path as osp
import subprocess

import numpy as np
from gym import error

class SegmentVideoRecorder(object):
    def __init__(self, predictor, env, save_dir, checkpoint_interval=500):
        self.predictor = predictor
        self.env = env
        self.checkpoint_interval = checkpoint_interval
        self.save_dir = save_dir

        self._num_paths_seen = 0  # Internal counter of how many paths we've seen
        self._counter = 0  # Internal counter of how many videos we've saved at a given iteration.

    def path_callback(self, path):
        if self._num_paths_seen % self.checkpoint_interval == 0:  # and self._num_paths_seen != 0:
            fname = '%s/run_%s_%s.mp4' % (self.save_dir, self._num_paths_seen, self._counter)
            print("Saving video of run %s_%s to %s" % (self._num_paths_seen, self._counter, fname))
            write_segment_to_video(path, fname, self.env)
        self._num_paths_seen += 1

        self.predictor.path_callback(path)

    def predict_reward(self, path):
        return self.predictor.predict_reward(path)

def write_segment_to_video(segment, fname, env):
    os.makedirs(osp.dirname(fname), exist_ok=True)
    frames = [env.render_full_obs(x) for x in segment["human_obs"]]
    for i in range(int(env.fps * 0.2)):
        frames.append(frames[-1])
    export_video(frames, fname, fps=env.fps)

def export_video(frames, fname, fps=10):
    assert "mp4" in fname, "Name requires .mp4 suffix"
    assert osp.isdir(osp.dirname(fname)), "%s must be a directory" % osp.dirname(fname)

    raw_image = isinstance(frames[0], tuple)
    shape = frames[0][0] if raw_image else frames[0].shape
    encoder = ImageEncoder(fname, shape, fps)
    for frame in frames:
        if raw_image:
            encoder.proc.stdin.write(frame[1])
        else:
            encoder.capture_frame(frame)
    encoder.close()

class ImageEncoder(object):
    def __init__(self, output_path, frame_shape, frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise Exception(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e. RGB values for a w-by-h image, with an optional alpha channl.".format(
                    frame_shape))
        self.wh = (w, h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec

        if distutils.spawn.find_executable('avconv') is not None:
            self.backend = 'avconv'
        elif distutils.spawn.find_executable('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise Exception(
                """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")

        self.start()

    @property
    def version_info(self):
        return {
            'backend': self.backend,
            'version': str(subprocess.check_output(
                [self.backend, '-version'],
                stderr=subprocess.STDOUT)),
            'cmdline': self.cmdline
        }

    def start(self):
        self.cmdline = (
            self.backend,
            '-nostats',
            '-loglevel', 'error',  # suppress warnings
            '-y',
            '-r', '%d' % self.frames_per_sec,
            '-f', 'rawvideo',  # input
            '-s:v', '{}x{}'.format(*self.wh),
            '-pix_fmt', ('rgb32' if self.includes_alpha else 'rgb24'),
            '-i', '-',  # this used to be /dev/stdin, which is not Windows-friendly
            '-vf', 'vflip',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            self.output_path
        )

        if hasattr(os, 'setsid'):  # setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise error.InvalidFrame(
                'Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
        if frame.shape != self.frame_shape:
            raise error.InvalidFrame(
                "Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(
                    frame.shape, self.frame_shape))
        if frame.dtype != np.uint8:
            raise error.InvalidFrame(
                "Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(frame.dtype))

        if distutils.version.LooseVersion(np.__version__) >= distutils.version.LooseVersion('1.9.0'):
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise Exception("VideoRecorder encoder exited with status {}".format(ret))

def upload_to_gcs(local_path, gcs_path):
    assert osp.isfile(local_path), "%s must be a file" % local_path
    assert gcs_path.startswith("gs://"), "%s must start with gs://" % gcs_path

    print("Copying media to %s in a background process" % gcs_path)
    subprocess.check_call(['gsutil', 'cp', local_path, gcs_path])
