import os
import time
import pyautogui as pg
import numpy as np
from dota_model import DotaBot
from time import gmtime, strftime
import pickle


class DotaGame:
  
  load_parameter = True
  is_train = False
  battle_field = (96, 978)
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
    ## reset the time
    self.bot.env.over_time = 0
    ## move to the battle field
    x, y = self.battle_field; pg.click(x, y, button="right")
    pg.PAUSE = 80
    pg.click(button="right")
    pg.PAUSE = tmp
    self.train()

  def train(self):
    iter_count = 0
    try:
      ## load the parameters if exist
      if self.load_parameter == True and os.path.isfile("train_parameters.npy"):
        self.load_parameter = False
        para = np.load("train_parameters.npy").item()
        self.bot.set_parameters(para)
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
        tmp = strftime("%d-%b-%Y", gmtime())
        with open('result_'+tmp+'-'+str(self.count), 'wb') as output:
          pickle.dump(self.bot.env.gold, output)
          pickle.dump(self.bot.env.reward, output)
        paras = self.bot.get_parameters()
        np.save('train_parameters', paras)
        
      self.relaunch()
    except KeyboardInterrupt:
      print("Done one training\n")
 
  def play(self):
    try:
      if self.load_parameter == True and os.path.isfile("train_parameters.npy"):
        para = np.load("train_parameters.npy").item()
        self.bot.set_parameters(para)
      while True:
        self.bot.onestep()
        self.bot.env.update()
        if self.is_end():
          print("Game End")
          return 0
    except KeyboardInterrupt:
      print("Done with the game \n")

  def is_end(self):
    pass

if __name__ == "__main__":
  time.sleep(3)
  game = DotaGame()
  DotaGame.load_parameter = True
  ## default is the play mode
  DotaGame.is_train = True
  if DotaGame.is_train == True:
    game.train()
  else:
    game.play()
