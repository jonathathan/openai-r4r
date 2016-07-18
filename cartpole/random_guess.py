import gym
import numpy as np

MAX_EPISODES = 20  # episodes per model
MAX_STEPS = 200  # max steps per episode
OUTPUT_DIR = '/tmp/cartpole-random-guess'

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
  env.monitor.start(OUTPUT_DIR)

  while True:
    w = np.random.rand(4) * 2 - 1  # uniform random [-1, 1)
    reward = sum(run_episode(env, w) for _ in xrange(MAX_EPISODES))
    if reward == MAX_EPISODES * MAX_STEPS:
      break

  # evaluate the best model for 100 episodes
  for _ in xrange(100):
    run_episode(env, w)

  env.monitor.close()
