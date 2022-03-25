[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[image2]: imgs/env.png "environment"

# Deep Q-Learning Project - Let's collect Bananas!

### Introduction

This repository explain how to train your own intelligent agent to catch bananas in a game. It's a project part of [Udacity Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) in Deep Reinforcement Learning.

The task selected to train our agents is Navigation. Basically, we have a 3d environment with lots of bananas (blue and yellow ones) and we need to try to get as much as possible yellow ones avoiding blue ones. Simple like that!

<center>

![Trained Agent][image1]

</center>

Focus on collecting yellow bananas, our environment gives reward of +1 when yellow bananas are collected and a reward of -1 when blue ones are collected. The state space has 37 dimensions and contains the ray-based perception of objects around agent's forward direction plus which objects are in this perception. Next figure illustrate 7 acquisitions of perception sensor.

<center>

![environment][image2]

</center>

The array of 7 acquisitions (state space) looks like this:

```python
[
 1.         0.         0.         0.         0.84408134
 0.         0.         1.         0.         0.0748472
 0.         1.         0.         0.         0.25755    
 1.         0.         0.         0.         0.74177343
 0.         1.         0.         0.         0.25854847
 0.         0.         1.         0.         0.09355672
 0.         1.         0.         0.         0.31969345
 0.         0.
]
```

First 4 elements represent the object (one-hot encoded of blue bananas, yellow bananas, wall and nothing) and the 5th element is the distance til the object (0 to 1). For actions we have 4:


    0 - walk forward
    1 - walk backward
    2 - turn left
    3 - turn right


### Getting Start

For first, let's clone this repository... \
(Let's assume that you are executing this on Linux OS)

1. Create a path to clone the project

```bash
mkdir NAME_OF_PROJECT & cd NAME_OF_PROJECT
```

2. Clone the project

```bash
git clone https://github.com/leocneves/dqn_navigation & cd dqn_navigation
```

3. Follow instructions in **Dependencies** from [THIS](https://github.com/udacity/Value-based-methods#dependencies) repository

4. Save file from [THIS LINK](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) in root of repository

5. Unzip it! (**Remember** this file has no vis, for see agent in environment download [THIS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip) version...)

```bash
sudo apt install unzip & unzip Banana_Linux_NoVis.zip
```

6. Done! In this repository we've 4 main files:

 - Navigation.ipynb: Notebook with train/test code;
 - dqn_agent.py: Code with all structure of our agent (Parameters, functions to step, select actions, update rules, etc...);
 - model.py: Contains the architecture of Neural Net applied to our agent;
 - Report.mb: Contains the description of code and results.


5. To train the agent just open the notebook **Navigation.ipynb** and execute all cells! At final of training step *(mean of last 100 rewards are more than +13 or episode are greater than 2000)* we can see *'checkpoint.pth'* created where contains the weights of neural nets from training step and in *'results/'* we can see graph generated to illustrate convergence in learning by plotting scores for each 100 episodes.
