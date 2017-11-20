import numpy as np
import pyautogui as pg
from scipy.sparse import csr_matrix
from scipy.signal import convolve2d

pg.PAUSE = 0
pg.FAILSAFE = True

_width, _height = pg.size()

## put hero in the center of the camera
def center_hero():
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


## Dota 2 environment
## features, the current state and environmental parameters for learning
class DotaEnv:
  TIME_LIMIT = 600
  ## time interval between two actions by pyautogui
  ## set true to raise an exception at (0, 0)
  over_time = None  # time in the game

  def __init__(self):
    center_hero()
    self.views = [np.array(pg.screenshot())]
    tmp = pg.PAUSE
    pg.PAUSE = 0.1
    self.views.append(np.array(pg.screenshot()))
    pg.PAUSE = tmp
    self.UI = DotaUI(self.views[-1])
    self.score = self.UI.get_score()
    self.reward = 0
    self.over_time = self.UI.check_time()

  ## update once the bot makes an action
  def update(self):
    ## screenshot corresponds to the action by the bot
    self.update_views()
    self.UI.update(self.views[-1])
    score = self.score
    self.score = self.UI.get_score()
    ## 10 is a magic number
    if self.score - score < 10 :  # marginal change does not count
      self.reward = 0
    else:
      self.reward = self.score - score

  def update_views(self):
    center_hero()
    self.views = [np.array(pg.screenshot())]
    tmp = pg.PAUSE
    pg.PAUSE = 0.1
    self.views.append(np.array(pg.screenshot()))
    pg.PAUSE = tmp


class DotaBot:
  MEMORY_LIMIT = 1000
  MEMORY_RETRIEVAL = 100
  def __init__(self):
    self.env = DotaEnv()
    self.policy = BotPolicy(self)
    self.memory = []

  ## interpret the commands and execute them
  def onestep(self, sess):
    center_hero()

    ## generate the commands based on the current state
    policy = self.policy
    X = policy.get_state()
    *command, meta = policy.forward()
    policy.execute(command)
    if len(self.memory) >= self.MEMORY_LIMIT:
      ## randomly throw away old record
      i = np.random.randint(len(self.memory) - self.RECENT_MEMORY)
      self.memory.pop(i)
    self.memory.append((X.copy(), command.copy()))
    return meta

  def get_parameters(self):
    return self.policy.paras

  def set_parameters(self, parameters):
    paras = self.get_parameters
    for k, v in parameters:
      paras[k] = v


class BotPolicy:
  BLACKPIXEL_PERCENT = 0.95
  def __init__(self, bot):
    self.bot = bot
    self.paras = {'w_conv1': None, 'w_fc1': None}
    self.kernel_size = 50
    self.paras['w_conv1'] = numpy.random.normal(loc=0, scale=0.05, size=[50, 50])
    self.paras['w_fc1'] = numpy.random.normal(loc=0, scale=0.05, \
                                              size=[_width, _height, 3])
    self.learning_rate = 1e-5
    self.sigma = 50 # magic number

  ## return the location of the click for a given state
  def forward(self, X):
    ## convolution
    w_conv1 = self.paras['w_conv1']
    conv1 = convolve2d(X, w_conv1, boundary='symm', mode='same')
    ## relu
    conv1[conv1 < 0] = 0
    ## fully connected layer
    w_fc1 = self.paras['w_fc1']
    X_conv1 = csr_matrix(conv1)
    fc1 = X_conv1.dot(w_fc1)
    x, y, z = fc1
    x = np.random.normal(x, scale=self.sigma)
    y = np.random.normal(y, scale=self.sigma)
    ## normalize to fit the screen size
    x = np.abs(x) % _width
    y = np.abs(y) % _height
    z = np.abs(z) % (_width + _height)
    ## store results for backpropogation
    mata = [x, y, z, X_conv1]
    return x, y, z, meta

  ## return the gradient of parameters
  def optimizer(self, meta):
    reward = self.bot.env.reward
    x, y, z, X_conv1 = meta
#    dw_fc1 = np.stack([2*x*X_conv1, 2*y*X_conv1, 2*z*X_conv1], axis=2) * reward
#    self.bot.paras['w_fc1'] -= dw_fc1 * self.bot.learning_rate
    dw_conv1 = numpy.random.normal(loc=0, scale=0.05, size=[50, 50])
    loss_before = self.loss()
    self.bot.paras['w_conv1'] -= dw_conv1 * self.bot.learning_rate * reward
    loss_after = self.loss()
    if loss_after > loss_before:
      self.bot.paras['w_conv1'] += 2*dw_conv1 * self.bot.learning_rate * reward

    dw_fc1 = numpy.random.normal(loc=0, scale=0.05, size=[_width, _height, 3])
    loss_before = self.loss()
    self.bot.paras['w_fc1'] -= dw_fc1 * self.bot.learning_rate * reward
    loss_after = self.loss()
    if loss_after > loss_before:
      self.bot.paras['w_fc1'] += 2*dw_fc1 * self.bot.learning_rate * reward
  
  def loss(self):
    l = np.min(len(self.memory), self.MEMORY_RETRIEVAL)
    reward = self.bot.env.reward
    loss = 0
    for i in range(-1, -(l+1), -1):
      X = self.memory[i][0]
      predict_x, predict_y, predict_z, meta = self.forward(X)
      observe_x, observe_y, observe_z = self.memory[i][1]
      x = (predict_x - observe_x) % _width
      y = (predict_y - observe_y) % _height
      if predict_z * observe_z < 0:
        z = predict_z - observe_z
      else:
        z = 0
      z = np.abs(z) % (_width + _height)
      loss += x**2 + y**2 + z**2
    loss /= l
    return reward * loss

   
  def get_state(self, view1, view2):
    ## use the difference
    X = view1 - view2
    X = np.mean(X[:, :, 0:3], axis=2)
    width_per_block, height_per_block = _width // 10, _height // 10

    for i in np.arange(0, _width, width_per_block):
      for j in np.arange(0, _height, height_per_block):
        i = int(i); j = int(j)
        block = X[i:i+width_per_block, j:j+height_per_block]
        ## set entire block to 0 if high percentage of pixels are 0
        if np.sum(block == 0) / (width_per_block * height_per_block) > self.BLACKPIXEL_PERCENT:
          X[i:i+width_per_block, j:j+height_per_block] = 0
    return X

  def execute(self, command):
    ## TODO: tune the parameter
    tmp = pg.PAUSE
    pg.PAUSE = 0.7
    button=['right' if command[2] > 0 else 'left']
    pg.click(x=command[0], y=command[1], button=botton)
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
    if np.sum(z > 7200) == 12:
      return 1
    else:
      return 0

  def get_score(self):
    gold = self.get_gold()
    ability_lvl = self.unlock_ability()
    lvl = self.get_lvl()
    score = gold * (lvl + ability_lvl)
    return score
