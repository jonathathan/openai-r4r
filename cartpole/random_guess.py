import gym
import numpy as np

MAX_EPISODES = 20  # episodes per model
MAX_STEPS    = 200 # max steps per episode
OUTPUT_DIR   = '/tmp/cartpole-random-guess'  # env monitor output goes here

def get_action(w, s):
  return int(np.dot(w, s) > 0)

def run_episode(env, w):
  ''' returns the total reward over the episode '''
  s = env.reset()
  done = False
  reward = 0.0
  for _ in xrange(MAX_STEPS):
    a = get_action(w, s)
    s, r, done, _ = env.step(a)
    reward += r
    if done:
      break
  return reward

if __name__ == '__main__':
  env = gym.make('CartPole-v0')

  for i in xrange(10000):
    w = np.random.rand(4) * 2 - 1  # uniform random [-1, 1)
    reward = sum(run_episode(env, w) for _ in xrange(MAX_EPISODES))
    print reward / MAX_EPISODES
