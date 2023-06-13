# Code for Group 2's project in AI Algorithmic Art class @ University of Warsaw

<font color=red>This is a WIP!</font>

This is the code for an art project that will modify [Nicklas Hansen](https://nicklashansen.github.io) et al.'s awesome **Temporal Difference-Model Predictive Control** algorithm for solving `dm_control`'s challenging Dog-Run task. We will be running experiments on the fully developed agent which can be called psychological in nature. 

See the original paper, [Temporal Difference Learning for Model Predictive Control](https://arxiv.org/abs/2203.04955) by [Nicklas Hansen](https://nicklashansen.github.io), [Xiaolong Wang](https://xiaolonw.github.io)\*, [Hao Su](https://cseweb.ucsd.edu/~haosu)\*

## Branches

This is the main, or `production` branch. Its goal is to serve as a laboratory for our experiments, as well as the main video production branch. This is what you want to use if you will be using CUDA.

For the purposes of testing out video generation and other things that can be achieved locally, please use the `videogen` branch. It can be run on a CPU (and in fact has CUDA support disabled; I'm going to make it configurable to choose CUDA or CPU in the future).

## License & Acknowledgements

Any code here is licensed under the MIT license. [TD-MPC](https://github.com/nicklashansen/tdmpc) is licensed under the MIT license. [MuJoCo](https://github.com/deepmind/mujoco) and [DeepMind Control Suite](https://github.com/deepmind/dm_control) are licensed under the Apache 2.0 license. We thank the [DrQv2](https://github.com/facebookresearch/drqv2) authors for their implementation of DMControl wrappers.
