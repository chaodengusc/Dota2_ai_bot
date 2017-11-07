import time
import pyautogui as pg
from dota_model import DotaBot

class DotaGame:
  MEMORY_LIMIT = 1000  # the maximum length of the track (state, action, reward)

  def __init__(self):
    self.bot = DotaBot()
    self.memory = []
    self.RECENT_MEMORY = 1

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
    self.train()

  def train(self):
    try:
      while not self.bot.env.over_time:
        self.bot.onestep()
        self.bot.env.update()
        reward = self.bot.env.reward
        if len(self.memory) >= self.MEMORY_LIMIT:
          ## randomly throw away old record
          i = np.random.randint(len(self.memory) - self.RECENT_MEMORY)
          self.memory.pop(i)
          self.memory.append((self.env.state, self.env.commands, reward))
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
  game.relaunch()
  game.train()
