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
    self.views = [np.array(pg.screenshot())]
    self.memory = []
    self.UI = DotaUI(self.views[0])
    self.score = self.UI.get_score()
    self.reward = 0
    time = self.UI.get_time()
    self.RECENT_MEMORY = 1 # things happened recently

  ## update once the bot makes an action
  def update(self):
    ## screenshot corresponds to the action by the bot
    self.view = np.array(pg.screenshot())
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
    if len(self.views) >= self.VIEWS_LIMIT:
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
      for i in np.random.randint(2, size=3)]
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

  def get_UI(self):
    return self.env.UI


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

  ## regions
  HP_REGION = tuple([(i, 1020, 12, 20) for i in range(984, 943, -10)])
  ## the digit and its feature for hp
  ## (0, 0) means no digit in the figure
#  HP_DIGITS = {(16, 18):0, (9, 6):1, (11, 17):2, (10, 15):3, (8:18):4, 
#              (12, 16):5, (10, 18):6, (12, 7):7, (16, 19):8, (15, 12):9, (0, 0):0}
  HP_DIGITS = dict(zip([75030, 69872, 76838, 77303, 74662, 76570, 79967, 
                        71074, 82300, 78704], range(10)))

  LVL_REGION = (598, 1046, 24, 20)
  ## the digit and its feature for lvl
  LVL_DIGITS = dict(zip([34514, 50320, 47276, 47593, 49772, 53297, 40642, 59439,
                         51928, 85645, 67183, 80362, 78174, 77916, 80040, 82476,
                         71740, 88125, 81374, 98408, 81196, 93589, 91638, 91665,
                         93173], range(1, 26)))

  TIME_REGION = tuple([(i, 22, 9, 18) for i in (970, 962, 949, 941)])
  ## the digit and its feature for time
  TIME_DIGITS = dict(zip([51460, 43591, 48149, 46679, 47013, 49132, 49033,
                          44022, 52326, 49675], range(10)))


  def __init__(self, view):
    self.view = view
    self.width, self.height = pg.size()

  def update(self, view):
    self.view = view

  def get_hp(self):
    digits = []
    for i in self.HP_REGION:
      region = self.view[i[0]:i[1],i[0]+i[2]:i[1]+i[3], 0:3]
#      z = np.sum(region, axis=2)
#      f1 = np.sum(z[0:z.shape[0]//2, :]==765)
#      f2 = np.sum(z[z.shape[0]//2:z.shape[0] , :]==765)
#      digits.apend(HP_DIGITS[(f1, f2)])
      z = np.sum(region)
      if z in self.HP_DIGITS:
        digits.apend(self.HP_DIGITS[z])
      else:
        digits.apend(0)

    num = 0
    for i in range(len(digits)):
      num += 10^i * digits[i]
    return num

  def get_gold(self):
    ## TODO
    return 1

  def get_lvl(self):
    i = self.LVL_REGION
    region = self.view[i[0]:i[1],i[0]+i[2]:i[1]+i[3], 0:3]
    z = np.sum(region)
    if z in self.LVL_DIGITS:
      lvl = self.LVL_DIGITS[z]
    else:
      lvl = 0
    return lvl

  def get_ability_lvl(self):
    ## TODO
    return 1
  
  def get_time(self):
    digits = []
    for i in self.TIME_REGION:
      region = self.view[i[0]:i[1],i[0]+i[2]:i[1]+i[3], 0:3]
      z = np.sum(region)
      if z in self.TIME_DIGITS:
        digits.apend(self.TIME_DIGITS[(f1, f2)])
      else:
        digits.append(0)

    time = digits[0] + digits[1] * 10
    time += (digits[2] + digits[3] * 10) * 60
    return time

  def get_score(self):
    gold = self.get_gold()
    ability_lvl = self.get_ability_lvl()
    lvl = self.get_lvl()
    score = gold * (lvl + ability_lvl)
    return score
