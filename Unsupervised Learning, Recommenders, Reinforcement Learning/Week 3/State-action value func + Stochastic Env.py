import numpy as np

# Do not modify
num_states = 6
num_actions = 2

terminal_left_reward = 100
terminal_right_reward = 40
each_step_reward = 0

# Discount factor
gamma = 0.5

# Probability of going in the wrong direction
misstep_prob = 0 # Change to 0.1 to apply stochastic environment where the agent can go in the wrong direction
