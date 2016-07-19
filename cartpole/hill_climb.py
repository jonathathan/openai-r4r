import gym
import numpy as np

CUTOFF_STREAK = 1000  # stop when we haven't seen improvement in 1000 episodes
MAX_EPISODES  = 10    # number of episodes to evaluate each model
MAX_STEPS     = 200   # max steps per episode
MAX_NOISE     = 0.05  # max amount of noise (as %) to add per iteration
OUTPUT_DIR    = '/tmp/cartpole-random-guess'  # env monitor output goes here

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

  # base model: uniform random [-1, 1)
  w_best = np.random.rand(4) * 2 - 1
  r_best = sum(run_episode(env, w_best) for _ in xrange(MAX_EPISODES))
  streak = 0

  while streak < CUTOFF_STREAK:
    noise = np.random.rand(4) * 2 * MAX_NOISE - MAX_NOISE
    w_new = w_best + noise
    r_new = sum(run_episode(env, w_new) for _ in xrange(MAX_EPISODES))

    if r_new > r_best:
      print 'new pb', r_new, 'after', streak
      w_best, r_best = w_new, r_new
      streak = 0
    else:
      streak += 1

  print 'final', r_best
