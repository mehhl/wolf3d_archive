import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
import env as environment
from dm_control.suite.wrappers import action_scale, pixels
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


import hide

__CONFIG__, __LOGS__ = 'cfgs', 'logs'

_LEARNER_ACT_SPEC = 38
_SEEKER_ACT_SPEC = 12
_COUNT_SEEKERS = 3
_XSIZE=500
_YSIZE=500
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def actuate(env, obs, step, learner, seekers, l_t0, s_t0):
    """Actuates a suitable `Environment` with action from TDMPC object
    `learner` and an array `seekers` of TDMPC objects.

    Args:
        `env` -- a `dm_control.rl.control.Environment` and
        similar object, with observation dim 235+(k*78),
        action dim 38+(k*12), reward dim k+1.
        `obs` -- an array of 235+(k*78) observations from `env`.
        `learner` -- an instance of `algorithm.tdmpc.TDMPC`
        built from a suitable config (235 obs, 38 actions).
        `step` -- number of steps.
        `seekers` -- an iterable of k instances of `algorithm.tdmpc.TDMPC`
        built from a suitable config (78 obs, 12 actions) compatible
        for concurrent use with `learner`.
        `l_episode`, `s_episode` -- Episodes corresponding to
        `learner` and `seekers`."""
    _s_action_range = lambda i: np.arange(225+78*i, 225+78*(i+1))
    _s_plan = lambda i: seekers[i].plan(obs[_s_action_range(i)], step=step,
                                        t0 = s_t0)
    l_action = learner.plan(obs[0:225], step=step, t0=l_t0)
    s_actions = [_s_plan(i) for i in range(len(seekers))]
    actions = torch.cat([l_action] + s_actions)
    return (l_action, *env.step(actions.cpu().numpy()))


def evaluate(env, learner, seekers, num_episodes, step, env_step, video, ep_length):
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video: 
            video.init(env, enabled=(i==0))
        j = 0
        while j < ep_length:
            _, obs, reward, done, _ = actuate(env, obs, step, 
                                           learner, seekers,
                                           t==0, t==0)
            ep_reward += reward[0]
            if video: video.record(env, camera_id=2)
            t += 1
            j += 1
        episode_rewards.append(ep_reward)
        if video: video.save(env_step)
    return np.nanmean(episode_rewards)



def train():
    # read proper cfgs. this whole section is very ad hoc, TODO:formalize
    # read paths (fixed for now)
    l_fp = Path().cwd() / '..' / 'cfgs' / 'learner_cfg'
    s_fp = Path().cwd() / '..' / 'cfgs' / 'seekers_cfg' 
    # parse cfgs from paths
    l_cfg = parse_cfg(l_fp)
    s_cfg = parse_cfg(s_fp)

    # ad hoc fixes to cfgs
    s_cfg.episode_length = 500
    l_cfg.episode_length = 500
    l_cfg.obs_shape = [225]
    l_cfg.action_dim = 38
    s_cfg.obs_shape = [78]
    s_cfg.action_dim = 12
    l_cfg.train_steps = 250000
    s_cfg.action_repeat = 2

    # build env
    env = hide.hide(count_seekers=_COUNT_SEEKERS, xsize=_XSIZE, ysize=_YSIZE)
    env = environment.ActionDTypeWrapper(env, np.float32)
    env = environment.ActionRepeatWrapper(env, l_cfg.action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
    env = environment.ExtendedTimeStepWrapper(env)
    env = environment.TimeStepToGymWrapper(env, 'dog','hide', 
                                           l_cfg.action_repeat, 'state')
    env = environment.DefaultDictWrapper(env)
    
    # build learner and his replay buffer
    learner = TDMPC(l_cfg)
    l_replaybuffer = ReplayBuffer(l_cfg)
    # build seekers and their replay buffers
    seekers = [TDMPC(s_cfg) for i in range(_COUNT_SEEKERS)]
    # s_replaybuffers = [ReplayBuffer(s_cfg) for i in range(_COUNT_SEEKERS)]
    _s_action_range = lambda i: np.arange(225+78*i, 225+78*(i+1))
      
 
    L = logger.Logger(Path().cwd() / 'final_run_log', l_cfg)
    episode_idx, start_time = 0, time.time()
    learner.load(Path().cwd() / '..' / 'weights' / f'l_enriched.pt')
    for seeker in seekers:
        seeker.load(Path().cwd() / '..' / 'weights' / f'final_seeker_model.pt')
    obs = env.reset()
    for step in range(0, 
                      l_cfg.train_steps+l_cfg.episode_length, 
                      l_cfg.episode_length):
        print(step)
        obs = env.reset()
        _s_action_range = lambda i: np.arange(225+78*i, 225+78*(i+1))
        l_episode = Episode(l_cfg, obs[0:225])
        s_episodes = [Episode(s_cfg, obs[_s_action_range(i)]) 
                      for i in range(_COUNT_SEEKERS)]
        i = 0
        while i < l_cfg.episode_length:
            print(i)
            l_action, obs, reward, done, _ = actuate(env, obs, step, 
                                           learner, seekers,
                                           l_episode, s_episodes[0])
            l_episode += (obs[0:225], l_action, reward[0], done)
            print(reward[0])
            i += 1
        assert len(l_episode) == l_cfg.episode_length
        l_replaybuffer += l_episode

        # backpropagate the models
        train_metrics = {}
        if step >= l_cfg.seed_steps:
            num_updates = l_cfg.seed_steps if step == l_cfg.seed_steps else l_cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(learner.update(buffer, step+i))

        episode_idx += 1
        env_step = int(step*l_cfg.action_repeat)
        common_metrics = {
                'episode': episode_idx,
                'step': step,
                'env_step': env_step,
                'total_time': time.time() - start_time,
                'episode_reward': l_episode.cumulative_reward}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        if env_step % l_cfg.eval_freq == 0:
            common_metrics['episode_reward'] = evaluate(env, learner, seekers, 
                                                        l_cfg.eval_episodes,
                                                        step, env_step, L.video,
                                                        l_cfg.episode_length)

            L.log(common_metrics, category='eval')

train() 

