# RL-Teacher

`rl-teacher` is an implementation of [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741) [Christiano et al., 2017].

The system allows you to teach a reinforcement learning agent novel behaviors, even when both:

1. The behavior does not have a pre-defined reward function
2. A human can recognize the desired behavior, but cannot demonstrate it

It's also just a lot of fun to train simulated robots to do whatever you want! For example, in the MuJoCo "Walker" environment, the agent is usually rewarded for moving forwards, but you might want to teach it to do ballet instead:

<p align="center">
<img src="https://user-images.githubusercontent.com/306655/28396526-d4ce6334-6cb0-11e7-825c-63a85c8ff533.gif" />
</p>

See our [agent circus](#agent-circus) for other tricks that you can train an agent to do using `rl-teacher`

## What's in this repository?

- A [reward predictor](/rl_teacher/teach.py) that can be plugged into any agent, and learns to predict which actions the human teacher would approve of
- Several [example agents](/agents) that learn via a function specified by the reward predictor
- A [webapp](/human-feedback-api) that humans can use to give feedback, providing the data used to train the reward predictor. The webapp interface is shown below.

<p align="center">
<img src="https://user-images.githubusercontent.com/306655/28898662-f3cd9142-779b-11e7-9252-137f9107c099.gif" />
</p>

Together with a set of integrations to [OpenAI gym](https://github.com/openai/gym), these components implement the full system described in  [*Deep RL from Human Preferences*](https://arxiv.org/abs/1706.03741)

<p align="center">
<img src="https://blog.openai.com/content/images/2017/06/diagram-4.png" />
</p>


## Quick install

### Dependencies

> # pip dependencies
> Django ~= 3.2.16  
> dj_database_url ~= 1.0.0  
> whitenoise ~= 6.2.0  
> mujoco-py ~= 2.1.2.14  
> gym ~= 0.26.0  
> tensorflow >= 2.11.0
  
> # conda dependencies
> mesalib ~= 21.2.5
> glew ~= 2.1.0

### Installation

Obtain a license for [MuJoCo](https://www.roboti.us/license.html) and install the [binaries](https://www.roboti.us/index.html) on your system. For good documentation on MuJoCo installation, and an easy way to test that MuJoCo is working on your system, we recommend following the mujoco-py installation.

Set up a fresh `conda` environment that uses python 3.5.

Clone the `rl-teacher` repository anywhere you'd like. (For example, in `~/rl-teacher`).

Then run the following to install the rl-teacher code into your conda environment:

    cd ~/rl-teacher
    pip install -e .
    pip install -e human-feedback-api
    pip install -e agents/parallel-trpo[tf]
    pip install -e agents/pposgd-mpi[tf]

# Usage
## Baseline RL

Run the following command to do baseline reinforcement learning directly from the hard-coded reward function. This does not use human feedback at all, but is a good way to test that MuJoCo is working and that the RL agent is configured correctly and can learn successfully on its own.

    python rl_teacher/teach.py -p rl -e ShortHopper-v1 -n base-rl

By default, this will write tensorboard files to `~/tb/rl-teacher/base-rl`. Start tensorboard as follows:

    $ tensorboard --logdir ~/tb/rl-teacher/
    Starting TensorBoard b'47' at http://0.0.0.0:6006
    (Press CTRL+C to quit)

Navigate to http://0.0.0.0:6006 in a browser to view your learning curves, which should look like the following: 

![rl_graph](https://user-images.githubusercontent.com/306655/28930266-47bf7988-7827-11e7-907f-0d1e2d8a87f7.png)

## Synthetic labels

Next we'll use the two-part training scheme (train a separate reward predictor, and use RL on the predicted reward), but instead of collecting genuine human feedback, we'll generate synthetic feedback from the reward function hard-coded into the environment. This provides us with another sanity check and a useful comparison of learning from the reward predictor versus learning from the true reward.

Instead of `-p rl` above, we specify `-p synth` to use a synthetic predictor. We'll use the same environment (`-e ShortHopper-v1`), give this run a new name (`-n syn-1400`), and ask for 1400 total labels (`-l 1400`).

    python rl_teacher/teach.py -p synth -l 1400 -e ShortHopper-v1 -n syn-1400

Your tensorboard curves should look like the following (with learning from synthetic labels in brown):

![rl_and_synth_graph](https://user-images.githubusercontent.com/306655/28930393-ae011026-7827-11e7-9541-ca01c50c20ac.png)

If you'd like to know exactly how synthetic labels are calculated, you can read the code in `SyntheticComparisonCollector`. The system uses an exponentially decaying labeling rate that tangentially approaches the desired total number of labels:
![labeling_rate](https://user-images.githubusercontent.com/306655/28930442-d6b23c02-7827-11e7-817c-71e74d8a55df.png)

## Human labels

To train your agent based off of feedback from a real human, you’ll run two separate processes:

1. The agent training process. This is very similar to the commands we ran above.
2. A webapp, which will show you short video clips of trajectories and ask you to rate which clip is better.

#### Set up the `human-feedback-api` webapp
First you'll need to set up django. This will create a `db.sqlite3` in your local directory.

    python human-feedback-api/manage.py migrate
    python human-feedback-api/manage.py collectstatic

Start the webapp

    python human-feedback-api/manage.py runserver 0.0.0.0:8000

You should now be able to open the webapp by navigating to http://127.0.0.1:8000/ in any browser. There’s nothing there yet, but when you run your agent, it will create an experiment that will let you add labels.

#### Create a GCS bucket to store rendered trajectory segments
The training process generates rendered trajectory segments for you to provide feedback on. These are stored in Google Cloud Storage (GCS), so you will need to set up a GCS bucket.

Note: if you would like to upload trajectory segments from your local storage, skip this part and set [-b] to "local".

If you don't already have GCS set up, [create a new GCS account](https://cloud.google.com/storage/docs/) and set up a new project. Then, use the following commands to create a bucket to host your media and set this new bucket to be publicly-readable.

    export RL_TEACHER_GCS_BUCKET="gs://rl-teacher-<YOUR_NAME>"
    gsutil mb $RL_TEACHER_GCS_BUCKET
    gsutil defacl ch -u AllUsers:R $RL_TEACHER_GCS_BUCKET

#### Run your agent
Now we're ready to train an agent with human feedback!

Note: if you have access to a remote server, we highly recommend running the agent training remotely, and provide feedback in the webapp locally. You can run both the agent training and the feedback app on your local machine at the same time. However, it will be annoying, because the rendering process during training will often steal window focus. For more information on running the agent training remotely, see the [Remote Server instructions](#using-a-remote-server-for-agent-training) below.

Run the command below to start the agent training. The agent will start to take random actions in the environment, and will generate example trajectory segments for you to label:

    $ python rl_teacher/teach.py -p human --pretrain_labels 175 -e Reacher-v1 -n human-175 -b "gs://rl-teacher-<YOUR_NAME>"
    Using TensorFlow backend.
    No label limit given. We will request one label every few seconds
    Starting random rollouts to generate pretraining segments. No learning will take place...
    -------- Iteration 1 ----------
    Average sum of true rewards per episode:     -10.5385
    Entropy:                                     2.8379
    KL(old|new):                                 0.0000
    Surrogate loss:                              0.0000
    Frames gathered:                           392
    Frames gathered/second:                 213857
    Time spent gathering rollouts:               0.00
    Time spent updating weights:                 0.32
    Total time:                                  0.33
    Collected 10/875 segments
    Collected 20/875 segments
    Collected 30/875 segments
    ...

Once the training process has generated videos for the trajectories it wants you to label, you will see it uploading these to GCS:

    ...
    Copying media to gs://rl-teacher-catherio/d659f8b4-c701-4eab-8358-9bd532a1661b-right.mp4 in a background process
    Copying media to gs://rl-teacher-catherio/9ce75215-66e7-439d-98c9-39e636ebb8a4-left.mp4 in a background process
    ...

In the meantime the agent training will pause, and wait for your feedback:

    0/175 comparisons labeled. Please add labels w/ the human-feedback-api. Sleeping...


#### Provide feedback to agent

At this point, you can click on the active experiment to enter the labeling interface. Click the Active Experiment link.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_70DA235B532C29881237939A9B4710C579961BE380D0A4E3A7AD2910654AC547_1500511446319_Screen+Shot+2017-07-19+at+4.37.46+PM.png)


Once you are in the labeling interface, you will see pairs of clips. For each pair, indicate which one shows better behavior, for whatever you are trying to teach the agent to do. (To start with, you might try to teach Reacher how to spin counter-clockwise, or come up with your own task!)

![](https://d2mxuefqeaa7sj.cloudfront.net/s_70DA235B532C29881237939A9B4710C579961BE380D0A4E3A7AD2910654AC547_1500511630006_compare.gif)

Once you have finished labeling the 175 pretraining comparisons, we train the predictor to convergence on the initial comparisons. After that, it will request additional comparisons every few seconds.

If you see a blank screen like this at any point, it means the clip is not yet ready to display. Try waiting a few minutes and refreshing the page, or click `Can't tell` to move on and try another clip

That's it! The more feedback you provide, the better your agent will get at the task.


### Using a remote server for agent training

We recommend running the agent on a server with many CPUs in order to get faster training.

If you're running on a remote server, you may need to log into your gcloud account with `gcloud auth login`.

If you’re running on a linux server without a display, you should follow the headless video rendering instructions below. This is not simply to visualize the agent's progress - it is necessary to render the segments for human labeling.

#### Headless video rendering on linux with Xorg/XDummy

If you're running on a machine without a physical monitor, you'll need to install XDummy. The following instructions have been tested on Ubuntu 14.04 LTS.

Install requirements

    sudo apt-get update && sudo apt-get install -y \
        ffmpeg \
        libav-tools \
        libpq-dev \
        libjpeg-dev \
        cmake \
        swig \
        python-opengl \
        libboost-all-dev \
        libsdl2-dev \
        xpra

Install Xdummy

    curl -o /usr/bin/Xdummy https://gist.githubusercontent.com/nottombrown/ffa457f020f1c53a0105ce13e8c37303/raw/ff2bc2dcf1a69af141accd7b337434f074205b23/Xdummy
    chmod +x /usr/bin/Xdummy 

Start Xdummy on display `:0`

    Xdummy

Test that video rendering works end to end

    DISPLAY=:0 python rl_teacher/tests/video_render_test.py


# Agent Circus
On the right are agents that were trained to do tricks based off human feedback; on the left, their counterparts trained with traditional RL. All videos are cherry-picked. Cheetah was trained using PPO, and all other agents were trained with TRPO.

| Walker Normal  | Walker Ballerina               |
| -------------- | ---------------------------- |
| ![walker_normal](https://user-images.githubusercontent.com/306655/28396993-a45caa1e-6cb3-11e7-8f6c-c043ca76ff01.gif) | ![walker_ballerina](https://user-images.githubusercontent.com/306655/28396526-d4ce6334-6cb0-11e7-825c-63a85c8ff533.gif)|

| Reacher Normal  | Reacher Opposite               |
| -------------- | ---------------------------- |
| ![reacher_normal](https://user-images.githubusercontent.com/306655/28396465-646c8800-6cb0-11e7-8f15-beab00d46f03.gif) | ![reacher_opposite](https://user-images.githubusercontent.com/306655/28396509-a0e5ed76-6cb0-11e7-98d4-08b7cd905706.gif) |

| Hopper Normal  | Hopper Backflip              |
| -------------- | ---------------------------- |
| ![hopper_normal](https://user-images.githubusercontent.com/306655/28431273-a6e19666-6d38-11e7-8c96-cab779147546.gif) | ![hopper_backflip_full](https://user-images.githubusercontent.com/306655/28431444-420bb82e-6d39-11e7-84a5-6f9f93883031.gif) |


| Cheetah Normal  | Cheetah Tapdance              |
| -------------- | ---------------------------- |
| ![ppo_cheetah_normal](https://user-images.githubusercontent.com/306655/28886801-9682c7a4-776f-11e7-8c28-8dd5b7f9b0e9.gif) | ![ppo_cheetah_hindleg](https://user-images.githubusercontent.com/306655/28886748-6c4a30c6-776f-11e7-84c5-2314835fc207.gif) |

# Extensions to this work

[rl-teacher-atari](https://github.com/machine-intelligence/rl-teacher-atari) by ([@raelifin](https://github.com/Raelifin)) supports atari environments, and uses a red-black tree to more efficiently find an ordering of human preferrences over clips.

# Acknowledgments
A huge thanks to Paul Christiano and Dario Amodei for the design of this system and for encouragement to make an open source version.  

Max Harms ([@raelifin](https://github.com/Raelifin)) wrote substantial portions of the system. Max integrated and tuned the parellized TRPO implementation, added many additional features and improvements, and trained the picturesque Walker Ballerina featured prominently in this repo.

Also a big thanks to Catherine Olsson ([@catherio](https://github.com/catherio)) for immensely improving the documentation and usability of `rl-teacher`. And thanks to Kevin Frans ([@kvfrans](https://github.com/kvfrans)) for his fast open-source [parallel-trpo implementation](https://github.com/kvfrans/parallel-trpo).  


Thank you for all members of the Deepest season 12 who support this project.
