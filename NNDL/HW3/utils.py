import glob
import io
import base64
import os
import numpy as np

from gym.wrappers import Monitor

from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display



def show_videos():
  """
  Simple function to reproduce videos inside the jupyter notebook

  """
  mp4list = glob.glob('video/*.mp4')
  mp4list.sort()
  for mp4 in mp4list:
    print(f"\nShowing test {mp4[-6:-4]}")
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    
def wrap_env(env, video_callable=None):
  """
  Wrap environment to produce videos from each episode
  """
  env = Monitor(env, './video', force=True, video_callable=video_callable)
  return env

def exp_prof(in_value, iters, weight=6):
  """
  Customizable exploration profile based on initial value of temperature for a softmax distribution
  """
  exp_decay = np.exp(-np.log(in_value) / iters * weight) # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
  exploration_profile = [in_value * (exp_decay ** i) for i in range(iters)]

  return exploration_profile


def exp_prof_egr(in_value, iters, weight=6):
  """
  Customizable exploration profile based on initial value of epsilon for a epsilon-greedy exploration
  """
  exp_decay = np.exp(- weight / (iters)) # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
  exploration_profile = [in_value * (exp_decay ** i) for i in range(iters)]

  return exploration_profile

