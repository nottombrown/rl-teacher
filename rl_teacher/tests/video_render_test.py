import os.path as osp
import uuid

from rl_teacher.envs import make_with_torque_removed
from rl_teacher.segment_sampling import segments_from_rand_rollout
from rl_teacher.teach import CLIP_LENGTH
from rl_teacher.video import write_segment_to_video

TEST_RENDER_DIR = '/tmp/rl_teacher_media_test'

def test_render_videos():
    env_id = "Hopper-v1"
    env = make_with_torque_removed(env_id)
    segments = segments_from_rand_rollout(env_id, make_with_torque_removed,
        n_desired_segments=1, clip_length_in_seconds=CLIP_LENGTH)

    for idx, segment in enumerate(segments):
        local_path = osp.join(TEST_RENDER_DIR, 'test-%s.mp4' % idx)
        print("Writing segment to: %s" % local_path)
        write_segment_to_video(
            segment,
            fname=local_path,
            env=env)

if __name__ == '__main__':
    test_render_videos()
