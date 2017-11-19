import numpy as np
import pyautogui as pg
import tensorflow as tf

pg.PAUSE = 0
pg.FAILSAFE = True

## Dota 2 environment
## features, the current state and environmental parameters for learning
class DotaEnv:
  TIME_LIMIT = 600
  ## time interval between two actions by pyautogui
  ## set true to raise an exception at (0, 0)
  over_time = None  # time in the game

  def __init__(self):
    self.center_hero()
    self.views = [np.array(pg.screenshot())]
    tmp = pg.PAUSE
    pg.PAUSE = 0.1
    self.views.append(pg.screenshot())
    pg.PAUSE = tmp
    self.UI = DotaUI(self.views[-1])
    self.score = self.UI.get_score()
    self.reward = 0
    self.over_time = self.UI.check_time()

  ## update once the bot makes an action
  def update(self):
    ## screenshot corresponds to the action by the bot
    self.update_views(self.views)
    self.UI.update(self.views[-1])
    score = self.score
    self.score = self.UI.get_score()
    ## 10 is a magic number
    if self.score - score < 10 :  # marginal change does not count
      self.reward = 0
    else:
      self.reward = self.score - score

  def update_views(self):
    self.center_hero()
    self.views = [np.array(pg.screenshot())]
    tmp = pg.PAUSE
    pg.PAUSE = 0.1
    self.views.append(pg.screenshot())
    pg.PAUSE = tmp


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
    x = np.random.randint(UI.width, size=3)
    y = np.random.randint(UI.height, size=3)
    buttons = ['left' if i==0 else 'right' \
      for i in np.random.randint(2, size=3)]
    return [(x[i], y[i], buttons[i]) for i in range(3)]

  ## execute the commands
  def execute(self, commands):
    ## TODO: tune the parameter
    pg.PAUSE = 0.5
    for i in commands:
      pg.click(x=i[0], y=i[1], button=i[2])

  ## interpret the commands and execute them
  def onestep(self):
    UI = self.get_UI()
    self.center_hero()

    ## generate the commands based on the current state
    self.state = self.get_state()
    self.commands = self.policy()
    self.execute(self.commands)

  def get_parameters(self):
    ## TODO
    return self.parameters

  def set_parameters(self, parameters):
    ## TODO
    pass

  def get_state(self):
    return self.env.views[-1]

  def get_UI(self):
    return self.env.UI

  ## put hero in the center of the camera
  def center_hero(self):
    tmp = pg.PAUSE
    pg.PAUSE = 0
    for i in range(570, 820, 60):
      pg.click(x=i, y=20, button='left')
    for i in range(1095, 1345, 60):
      pg.click(x=i, y=20, button='left')
    pg.click(x=880, y=20, button='right')
    ## left click the picture of the hero in the UI to put the hero
    ## in the center of the camera
    HERO_PICTURE = (634, 1002)
    pg.PAUSE = 0.1
    pg.click(x=HERO_PICTURE[0], y=HERO_PICTURE[1], button='left')
    pg.PAUSE = tmp
    

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
  YES_DISCONNECT = (860, 632)
  LOCK_IN = (1473, 804)

  ## regions
  HP_REGION = tuple([(i, 1020, 12, 20) for i in range(920, 879, -10)])
  HP_DIGITS = {(16, 18):0, (9, 6):1, (11, 17):2, (10, 15):3, (8, 18):4, 
               (12, 16):5, (10, 18):6, (12, 7):7, (16, 19):8, (15, 12):9}
  
  ## the digit for game time; one unit means 10 mins 
  LVL_REGION = (598, 1046, 24, 20)
  ## the digit and its feature for lvl
  LVL_DIGITS = dict(zip([34514, 50320, 47276, 47593, 49772, 53297, 40642, 59439,
                         51928, 85645, 67183, 80362, 78174, 77916, 80040, 82476,
                         71740, 88125, 81374, 98408, 81196, 93589, 91638, 91665,
                         93173], range(1, 26)))

  ## the digit for minute
  TIME_REGION = (941, 22, 9, 18)

  ## ability point
  ABILITIES = [(i, 1009) for i in range(855, 889, 11)] + \
              [(i, 1009) for i in range(920, 954, 11)] + \
              [(i, 1009) for i in range(985, 1019, 11)] + \
              [(i, 1009) for i in range(1055, 1078, 11)]
  ## the color (RGB) for unlock an ability
  UNLOCK_ABILITY = (180, 162, 106)

  ## gold region
  GOLD_REGION = [(i, 1040, 13, 20) for i in range(1737, 1684, -13)]
  ## the digit and its featurs for gold
  GOLD_DIGITS = dict(zip([(16, 22, 15), (11, 12, 8), (15, 11, 20), (12, 13, 15),
                          (5, 21, 6), (10, 15, 15), (8, 21, 14), (12, 10, 8),
                          (16, 24, 16), (15, 22, 8)], range(10)))
  

  def __init__(self, view):
    self.view = view
    self.width, self.height = pg.size()

  def update(self, view):
    self.view = view

  def get_hp(self):
    digits = []
    for i in self.HP_REGION:
      region = self.view[i[1]:i[1]+i[3], i[0]:i[0]+i[2], 0:3]
      z = np.sum(region, axis=2)
      f1 = np.sum(z[0:z.shape[0]//2, :]==765)
      f2 = np.sum(z[z.shape[0]//2:z.shape[0] , :]==765)
      if (f1, f2) in self.HP_DIGITS:
        digits.append(self.HP_DIGITS[(f1, f2)])
      else:
        digits.append(0)

    num = 0
    for i in range(len(digits)):
      num += 10**i * digits[i]
    return num

  def get_gold(self):
    digits = []
    ## magic position
    indicator_spot = (1060, 1785)
    if np.sum(self.view[indicator_spot[0], indicator_spot[1], ]) == 440:
      pixel_color = 765
    else:
      pixel_color = 510
    for i in self.GOLD_REGION:
      region = self.view[i[1]:i[1]+i[3], i[0]:i[0]+i[2], 0:3]
      z = np.sum(region, axis=2)
      f1 = np.sum(z[0:z.shape[0]//3,]==pixel_color)
      f2 = np.sum(z[z.shape[0]//3:z.shape[0]//3*2,]==pixel_color)
      f3 = np.sum(z[z.shape[0]//3*2:z.shape[0], ]==pixel_color)
      if (f1, f2, f3) in self.GOLD_DIGITS:
        digits.append(self.GOLD_DIGITS[(f1, f2, f3)])
      else:
        digits.append(0)

    num = 0
    for i in range(len(digits)):
      num += 10**i * digits[i]
    return num

  def get_lvl(self):
    i = self.LVL_REGION
    region = self.view[i[1]:i[1]+i[3], i[0]:i[0]+i[2], 0:3]
    z = np.sum(region)
    if z in self.LVL_DIGITS:
      lvl = self.LVL_DIGITS[z]
    else:
      lvl = 0
    return lvl

  def unlock_ability(self):
    unlock_ability = [all(self.view[x, y, ]== self.UNLOCK_ABILITY) \
                      for y, x in self.ABILITIES]
    return np.sum(unlock_ability)
  
  ## check if the game time is over 10 mins
  def check_time(self):
    digits = []
    i = self.TIME_REGION
    region = self.view[i[1]:i[1]+i[3], i[0]:i[0]+i[2], 0:3]
    z = np.sum(region, axis=2)
    ## game has played for 10 mins
    if np.sum(z > 720) == 12:
      return 1
    else:
      return 0

  def get_score(self):
    gold = self.get_gold()
    ability_lvl = self.unlock_ability()
    lvl = self.get_lvl()
    score = gold * (lvl + ability_lvl)
    return score
