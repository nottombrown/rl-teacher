import os
import os.path as osp
import uuid

import rl_teacher.agent.trpo.run_trpo_mujoco
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.segment_sampling import RandomRolloutSegmentCollector
from rl_teacher.video import write_segment_to_video

def test_render_videos():
    env = make_with_torque_removed("Hopper-v1")
    collector = RandomRolloutSegmentCollector(20000, env=env)
    rl_teacher.agent.trpo.run_trpo_mujoco.train(
        num_timesteps=8000,
        env=env,
        seed=0,
        predictor=collector,
        random_rollout=True,
    )

    segments = collector.segments
    tmp_media_dir = '/tmp/rl_teacher_media_test'
    for segment in segments:
        local_path = osp.join(tmp_media_dir, str(uuid.uuid4()) + '.mp4')
        print("Writing segment to: %s" % local_path)
        write_segment_to_video(
            segment,
            fname=local_path,
            env=env)

if __name__ == '__main__':
    test_render_videos()
