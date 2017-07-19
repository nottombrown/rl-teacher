# parallel-trpo

A parallel implementation of Trust Region Policy Optimization on environments from OpenAI gym.

Now includes hyperparaemter adaptation as well! More more info, check [Kevin Frans' post on this project](http://kvfrans.com/speeding-up-trpo-through-parallelization-and-parameter-adaptation/).
The code is based off of [this implementation](https://github.com/ilyasu123/trpo).

Augmented and adapted by Max Harms for the rl-teacher environment.

How to run:
```
# This just runs a simple training on InvertedPendulum-v1.
python main.py
```
Parameters:
```
--env_id: The gym environment to learn. (Default=InvertedPendulum-v1)
--run_name: A tag to associate with the optimization run logs. (Default=test_run)
--workers: How many subprocesses to use for generating environment data. (Default=4)
--runtime: How many seconds to run the program for. A value of 0 will cause it to run indefinitely. (Default=1800)
--max_kl: Maximum KL divergence between new and old policy. (Default=0.001)
```
