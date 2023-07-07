import random
from IPython.display import clear_output
import numpy as np
import gym
from pymongo import MongoClient

#Connect to MongoDB and select the database and collection to store the Q-table
client = MongoClient()
db = client['rl']
collection = db['q_table']

env = gym.make("Taxi-v2").env
env.render()

env.reset() # reset environment to a new, random state
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
#Action Space Discrete(6)
#State Space Discrete(500)

state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state
env.render()

#Initialize the Q-table using the MongoDB collection
if collection.count_documents({}) > 0:
    q_table = np.array(list(collection.find()))[:, 1:]  # Retrieve the Q-table from the collection
else:
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

state_size = env.observation_space.n
action_size = env.action_space.n


"""Training the agent"""

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        #Update the Q-table in the MongoDB collection
        collection.replace_one({'_id': state}, {'_id': state, 'values': q_table[state].tolist()}, upsert=True)

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

"""Evaluate agent's performance after Q-learning"""

q_table = np.array(list(collection.find()))[:, 1:]
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

client.close()