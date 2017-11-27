import os
import time
import pyautogui as pg
import numpy as np
from dota_model import DotaBot
from time import gmtime, strftime
import pickle
import scipy.misc


class DotaGame:
  
  load_parameter = True
  is_train = False
  battle_field = (110, 971)
  wraith_band = (1247, 770)
  bracer = (1335, 770)
  iron_branch = (1292, 598)
  def __init__(self):
    self.bot = DotaBot()
    self.count = 1
    self.rewards = []
    self.golds = []
    self.scores = []

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
    x, y = UI.BREWMASTER; pg.click(x, y, button="left")
    x, y = UI.LOCK_IN; pg.click(x, y, button="left")
    pg.PAUSE = 10
    x, y = self.bracer; pg.click(x, y, button="right")
    pg.PAUSE = 1
    x, y = self.iron_branch; pg.click(x, y, button="right", clicks=3, interval=0.5)
    x, y = UI.SKIP_AHEAD; pg.click(x, y, button="left")
    ## reset the time
    self.bot.env.over_time = 0
    ## move to the battle field
    pg.PAUSE = 10
    x, y = self.battle_field; pg.click(x, y, button="right")
    pg.PAUSE = 95
    pg.click(button="right")
    pg.PAUSE = tmp
    x, y = UI.THIRD_ABILITY; pg.click(x, y, button="left")
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
	## let bot remember the reward
        self.bot.memory[-1].append(reward)
        ## instant reward
        self.bot.policy.local_optimizer()
        iter_count += 1
        if iter_count % 10 == 0:
          self.rewards.append(reward)
      
      ## reward at the end of game
      self.bot.policy.global_optimizer()
      self.golds.append(self.bot.env.gold)
      self.scores.append(self.bot.env.gold + 100 * self.bot.env.lvl)
      self.count += 1
      tmp = strftime("%d-%b-%Y", gmtime())
      scipy.misc.imsave('./screenshots/'+tmp+str(self.count)+'.jpg', self.bot.env.UI.view)
      if self.count % 5 == 0:
        with open('result_'+tmp+'-'+str(self.count), 'wb') as output:
          pickle.dump(self.golds, output)
          pickle.dump(self.rewards, output)
          pickle.dump(self.scores, output)
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
