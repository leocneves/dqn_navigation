[//]: # (Image References)

[image1]: results/solved.png  "Results"

### Learning Algorithm

For first, this repository shows how to solve "Banana Unity Game" with a Deep Q-Network strategy. For this task we are using the base model available in Lessons of Udacity Nanodegree Deep Reinforcement Learning. Let's check it!

#### The 'model.py' file

Here we define the Neural Network that will be used in next steps. This architecture is pretty simple... we have 3 layers fully-connected.

```python
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
```

**fc1** represents the first hidden layer with state space size as input. **fc2** is the second hidden layer and **fc3** is the output layer, with number of actions activation as output.

The activation function used is ReLu, described in torch as:

```python
x = F.relu(self.fc1(state))
x = F.relu(self.fc2(x))
```


#### The 'dqn_agent.py' file

This file has the class **Agent**, that describes our agent with interact functions, and class **ReplayBuffer** that is responsible to manipulate the interaction experiences of the agent.

For first, for we initiate the agent we need to entry some initial params:

```python
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
```

Here we need to specify the state space size (37 on this navigation task), the action size (we've 4 actions on banana env.) and the seed.

Now, we have 3 main functions...

1. Step Function: This function is responsible for receives state, action, reward, next_state, done *from unity environment*, ADD the new observation experience from the env. in memory buffer and call the function *learn* to update weights with the new values. We can see the peace of code below:

```python
def step(self, state, action, reward, next_state, done):
    # Save experience in replay memory
    self.memory.add(state, action, reward, next_state, done)

    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % UPDATE_EVERY
    if self.t_step == 0:
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
```

2. Act Function: This function is responsible for choose actions from observed state (input param). Here we have a Epsilon-greedy action selection when some times we take a aleatory action (exploration) and another part of time we choose the most optimal action from neural net (exploitation).
We can control exploration vs exploitation changing epsilon (eps) variable.
We can see the peace of code below:

```python
def act(self, state, eps=0.):
    """Returns actions for given state as per current policy.

    Params
    ======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    self.qnetwork_local.eval()
    with torch.no_grad():
        action_values = self.qnetwork_local(state)
    self.qnetwork_local.train()

    # Epsilon-greedy action selection
    if random.random() > eps:
        return np.argmax(action_values.cpu().data.numpy())
    else:
        return random.choice(np.arange(self.action_size))
```

3. Learn Function: This function is responsible for update value parameters using given batch of experience tuples. In this step we basically update the neural net weights with batch of experiences. It's important to remember that we have two neural nets here with identical architectures, one for target model for current states and the second to compute Q values from local model.
We can see the peace of code below:

```python
def learn(self, experiences, gamma):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones = experiences

    # Get max predicted Q values (for next states) from target model
    Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    # Compute Q targets for current states
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

    # Get expected Q values from local model
    Q_expected = self.qnetwork_local(states).gather(1, actions)

    # Compute loss
    loss = F.mse_loss(Q_expected, Q_targets)
    # Minimize the loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # ------------------- update target network ------------------- #
    self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
```

---

The class **ReplayBuffer** just manipulate the experience buffer at each step by the agent. The code below shows the initialize parameters for this object.

```python
"""Params
    action_size (int): dimension of each action
    buffer_size (int): maximum size of buffer
    batch_size (int): size of each training batch
    seed (int): random seed
"""
self.action_size = action_size
self.memory = deque(maxlen=buffer_size)
self.batch_size = batch_size
self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
self.seed = random.seed(seed)    
```

Here we have to define the number of actions, memory (length of experiences stored), batch size to store in buffer and this class has two simple methods: add and sample.

The **add function** is responsible to append new experiences in the buffer as we can see in next code part (simple like that!):

```python
def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)   
```

The **sample function** is responsible to randomly sample a batch of experiences from memory and return this experiences to train the neural net.

```python
def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

    return (states, actions, rewards, next_states, dones)  
```

### Plot of Rewards

In the next figure we can see the plot of Score x Episodes til the end of episodes or the agent solve the task (Mean of last 100 scores equal or greater than +13.0).

![image1]

We can see that our agent was able to growth the reward received, showing the ability to learn with this env. and getting the score suggested by Udacity to declare the environment solved (more than +13.0) at episode **376**!

### Ideas for Future Work

For future work we can implement algorithms like **Prioritized Experience Replay**, **Double Q-Learning** and **Dueling Network Architectures**. Not just it, we can change neural nets architectures and RL parameters with grid search strategies!

---
