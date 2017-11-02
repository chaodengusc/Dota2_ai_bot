import numpy as np
import pyautogui as pg

## Dota 2 environment
## features, the current state and environmental parameters for learning
class DotaEnv:
  TIME_LIMIT = 600
  ## time interval between two actions by pyautogui
  ## TODO: tune this parameter
  pg.PAUSE = 0.1
  ## set true to raise an exception at (0, 0)
  pg.FAILSAFE = False
  time = None  # time in the game

  VIEWS_LIMIT = 100  # the maximum number of screenshots

  def __init__(self):
    self.view = None
    self.views = [pg.screenshot()]
    self.memory = []
    self.UI = DotaUI(self.views[0])
    self.score = self.UI.get_score()
    self.reward = 0
    time = self.UI.get_time()
    self.RECENT_MEMORY = 1 # things happend recently

  ## update once the bot makes an action
  def update(self):
    ## screenshot corresponds to the action by the bot
    self.view = pg.screenshot()
    self.UI.update(self.view)
    self.update_views(self.views)
    score = self.score
    self.score = self.UI.get_score()
    ## 10 is a magic number
    if self.score - score < 10 :  # marginal change does not count
      self.reward = 0
    else:
      self.reward = self.score - score
    self.time = self.UI.get_time()

  def update_views(self, view):
    if len(self.views) >= VIEWS_LIMIT:
      ## randomly remove an old view, but not the very recent ones
      ## RECENT_ONES = 0 means every view is considered old
      i = np.random.randint(len(self.memory) - self.RECENT_MEMORY)
      self.views.pop(i)
    self.views.append(view)


class DotaBot:
  def __init__(self):
    self.env = DotaEnv()
    self.parameters = {}
    self.state = None
    self.commands = None

  ## generate commands for the bot based on the current state
  def policy(self):
    ## TODO
    UI = self.get_UI()
    ## magic number 3 indicates a large operation space
    x = np.random.randint(UI.weight, size=3)
    y = np.random.randint(UI.height, size=3)
    buttons = ['left' if i==0 else 'right' \
      for i in np.random.randint(2, size=3)
    return [(x[i], y[i], buttons[i]) for i in range(3)]

  ## execute the commands
  def execute(self, commands):
    for i in commands:
      pg.click(x=i[0], y=i[1], button=i[2])

  ## interpret the commands and execute them
  def onestep(self):
    UI = self.get_UI()
    ## right click to select the hero
    x, y = pg.position()
    pg.click(x=x, y=y, button='right')
    ## left click the picture of the hero in the UI to put the hero
    ## in the center of the camera
    x, y = UI.hero_picture
    pg.click(x=x, y=x, button='left')

    ## generate the commands based on the current state
    self.state = self.get_state()
    self.commands = self.policy()
    self.execute(self.commands)

  def get_parameters(self):
    return self.parameters

  def set_parameters(self, parameters):
    pass

  def get_state(self):
    return self.env.views[-1]


class DotaUI:
  ## coordinates of key components in Dota 2
  CONTINUE = (956, 904)
  PLAY_DOTA = (1696, 1034)
  CREATE_LOBBY = (1652, 393)
  START_GAME = PLAY_DOTA
  MIRANA = (992, 417)
  SKIP_AHEAD = (163, 791)
  BACK_TO_DASHBOARD = (31, 27)
  DISCONNECT = (1676, 1036)
  LOCK_IN = (1473, 804)

  def __init__(self, view):
    self.view = view
    self.width, self.height = pyautogui.size()

  def update(self, view):
    self.view = view

  def get_hp(self):
    pass

  def get_gold(self):
    pass

  def get_lvl(self):
    pass

  def get_ability_lvl(self):
    pass
  
  def get_time(self):
    pass

  def get_score(self):
    gold = UI.get_gold()
    ability_lvl = UI.get_ability_lvl()
    lvl = UI.get_lvl()
    score = gold * (lvl + ability_lvl)
    return score
