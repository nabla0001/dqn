# Deep Q-Learning

My objective for this project was to replicate DeepMind's Deep Q-Network agent as published in [1]. 

This repo contains my implementation in `Python` using `tensorflow`  and OpenAI's `gym`.

[1] [Mnih et al. 2015. "Human-level control through Deep Reinforcement Learning". Nature 518.](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)


![Pong.gif](https://github.com/compgi13/assignment-3-reinforcement-learning-PrincipalComponent/blob/master/videos/Pong.gif)
![Boxing.gif](https://github.com/compgi13/assignment-3-reinforcement-learning-PrincipalComponent/blob/master/videos/Boxing.gif)
![Pacman.gif](https://github.com/compgi13/assignment-3-reinforcement-learning-PrincipalComponent/blob/master/videos/MsPacman.gif)

## Environment

`python 2.7`

`tensorflow 1.0`

`gym 0.8.1`

`tqdm 4.11.2` (simply `pip install tqdm` if missing)

## Folder structure

    /code     code base
    /logs     tensorboard logs and checkpoints
    /models   saved models

## Training

The agent be trained on any `atari` environment by running

     python main.py --env [atari-environment] # e.g. Pong-v3

OpenAI has a [list of all available atari games](https://gym.openai.com/envs#atari). The only requirement is that you specify  preprocessing options in `config.py` for any new environment.

All hyperparameters such as the learning rate for the Q-network or the exploration rate of the agent can be changed in `main.py`.

## Loading trained agents

The final checkpoints for all trained agents are saved in `/models`. Each agent can be loaded and its control performance
evaluated (by default using 20 episodes).

To load an agent go to the folder `./code` and run:

     python evaluate.py --env Pong-v3 --n_episodes 20          # Pong agent
     python evaluate.py --env Boxing-v3 --n_episodes 20        # Boxing agent

where `--env` can be any `gym` environment.


