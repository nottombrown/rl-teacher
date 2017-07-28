import argparse
import time

from parallel_trpo.train import train_parallel_trpo

from rl_teacher.summaries import make_summary_writer

class TraditionalRLRewardPredictor():
    """Always returns the true reward provided by the environment."""

    def predict_reward(self, path):
        return path["original_rewards"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRPO')
    parser.add_argument("-e", "--env_id", type=str, default='InvertedPendulum-v1')
    parser.add_argument("--run_name", type=str, default='test_run')
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--runtime", type=int, default=1800, help="The number of seconds to run for")
    parser.add_argument("--max_kl", type=float, default=0.001)
    args = parser.parse_args()

    env_id = args.env_id
    run_name = "%s/%s-%s" % (env_id, args.run_name, int(time.time()))

    summary_writer = make_summary_writer(run_name)

    train_parallel_trpo(
        env_id=args.env_id,
        run_name=args.run_name,
        predictor=TraditionalRLRewardPredictor(),
        summary_writer=summary_writer,
        workers=args.workers,
        runtime=args.runtime,
        max_kl=args.max_kl,
    )
