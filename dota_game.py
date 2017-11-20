import os
import time
import pyautogui as pg
import numpy as np
from dota_model import DotaBot
from time import gmtime, strftime


class DotaGame:
  
  load_parameter = True
  def __init__(self):
    self.bot = DotaBot()
    self.count = 1
    self.rewards = []
    self.golds = []

  def relaunch(self):
    tmp = pg.PAUSE
    UI = self.bot.get_UI()
    pg.PAUSE = 2
    x, y = UI.BACK_TO_DASHBOARD; pg.click(x, y, button="left")
    x, y = UI.DISCONNECT; pg.click(x, y, button="left")
    x, y = UI.YES_DISCONNECT; pg.click(x, y, button="left")
    x, y = UI.PLAY_DOTA; pg.click(x, y, button="left")
    x, y = UI.CREATE_LOBBY; pg.click(x, y, button="left")
    pg.PAUSE = 12
    x, y = UI.START_GAME; pg.click(x, y, button="left")
    pg.PAUSE = 2
    x, y = UI.MIRANA; pg.click(x, y, button="left")
    x, y = UI.LOCK_IN; pg.click(x, y, button="left")
    pg.PAUSE = 10
    x, y = UI.SKIP_AHEAD; pg.click(x, y, button="left")
    pg.PAUSE = tmp
    ## reset the time
    self.bot.env.over_time = 0
    self.train()

  def train(self):
    iter_count = 0
    try:
      if self.load_parameter == True and os.path.isfile("train_parameters.npz"):
        para = np.load("train_parameters.npz").item()
        self.bot.set_parameters(para)
      policy = self.bot.policy
      while not self.bot.env.over_time:
        meta = self.bot.onestep()
        self.bot.env.update()
        reward = self.bot.env.reward
        if reward != 0:
          self.bot.policy.optimizer(meta)
        iter_count += 1
        if iter_count % 1000 == 0:
          self.reward.append(reward)
      self.golds.append(self.bot.env.gold)
      self.count += 1
      if self.count % 5 == 0:
        tmp = strftime("%d-%b-%Y-%H:%M:%S", gmtime())
        with open('result'+tmp+'txt', 'wb') as output:
          pickle.dump(self.reward, self.gold)
        paras = self.bot.get_parameters()
        np.save('train_parameters.npz', paras)
        
      self.relaunch()
    except KeyboardInterrupt:
      print("Done one training\n")
 
  def play(self):
    try:
      while True:
        self.bot.onestep()
        self.bot.env.update()
    except KeyboardInterrupt:
      print("Done one game \n")


if __name__ == "__main__":
  time.sleep(3)
  game = DotaGame()
  game.train()
