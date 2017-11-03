import time
import pyautogui as pg

class DotaGame:
  MEMORY_LIMIT = 1000  # the maximum length of the track (state, action, reward)

  def __init__(self):
    self.bot = DotaBot()
    self.memory = []
    self.RECENT_MEMORY = 1

  def relaunch(self):
    UI = self.bot.UI
    time.sleep(20)
    x, y = BACK_TO_DASHBOARD; pg.click(x, y, button="left")
    time.sleep(10)
    x, y = DISCONNECT; pg.click(x, y, button="left")
    time.sleep(10)
    x, y = PLAY_DOTA; pg.click(x, y, button="left")
    time.sleep(5)
    x, y = CREATE_LOBBY; pg.click(x, y, button="left")
    time.sleep(5)
    x, y = START_GAME; pg.click(x, y, button="left")
    time.sleep(30)
    x, y = MIRANA; pg.click(x, y, button="left")
    time.sleep(10)
    x, y = SKIP_AHEAD; pg.click(x, y, button="left")

  def train(self):
    try:
      time = self.env.get_time()
      while time < self.env.MEMORY_LIMIT:
        self.bot.onestep()
        self.env.update()
        reward = self.env.reward
        if len(self.memory) >= MEMORY_LIMIT:
          ## randomly throw away old record
          i = np.random.randint(len(self.memory) - self.RECENT_MEMORY)
          self.memory.pop(i)
          self.memory.append((self.env.state, self.env.commands, reward))
      self.relaunch()
    except KeyboardInterrupt:
      print("Done one game \n")
 

    if len(self.memory) >= MEMORY_LIMIT:
      ## randomly throw away old record
      i = np.random.randint(len(self.memory) - self.RECENT_MEMORY)
      self.memory.pop(i)
      self.memory.append((self.env.state, self.env.commands, reward))

  def play(self):
    try:
      while True:
        self.bot.onestep()
        self.env.update()
    except KeyboardInterrupt:
      print("Done one game \n")


if __name__ == "__main__":
  game = DotaGame()
  game.play()
