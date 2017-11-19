import time
import pyautogui as pg
import tensorflow as tf
from dota_model import DotaBot

class DotaGame:

  def __init__(self):
    self.bot = DotaBot()

  def relaunch(self):
    tmp = pg.PAUSE
    UI = self.bot.get_UI()
    pg.PAUSE = 2
    x, y = UI.BACK_TO_DASHBOARD; pg.click(x, y, button="left")
    x, y = UI.DISCONNECT; pg.click(x, y, button="left")
    x, y = UI.YES_DISCONNECT; pg.click(x, y, button="left")
    x, y = UI.PLAY_DOTA; pg.click(x, y, button="left")
    x, y = UI.CREATE_LOBBY; pg.click(x, y, button="left")
    pg.PAUSE = 10
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
    try:
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      self.bot.policy.loss()
      self.bot.policy.optimizer()
      while not self.bot.env.over_time:
        self.bot.onestep(sess)
        self.bot.env.update()
        reward = self.bot.env.reward
        if reward != 0:
          sess(self.bot.policy.train_op)
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
