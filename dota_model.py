import numpy as np
import pyautogui as pg
from scipy.sparse import csr_matrix
from scipy.signal import convolve2d

pg.PAUSE = 0
pg.FAILSAFE = True

_width, _height = pg.size()

## put hero in the center of the camera
#def center_hero():
#  tmp = pg.PAUSE
#  pg.PAUSE = 0
#  for i in range(570, 820, 60):
#    pg.click(x=i, y=20, button='left')
#  for i in range(1095, 1345, 60):
#    pg.click(x=i, y=20, button='left')
#  pg.click(x=880, y=20, button='right')
#  ## left click the picture of the hero in the UI to put the hero
#  ## in the center of the camera
#  HERO_PICTURE = (634, 1002)
#  pg.PAUSE = 0.05
#  pg.click(x=HERO_PICTURE[0], y=HERO_PICTURE[1], button='left')
#  pg.PAUSE = tmp
def center_hero():
  pg.doubleClick(573, 22)


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
    pg.PAUSE = 0.05
    self.views.append(np.array(pg.screenshot()))
    pg.PAUSE = tmp
    self.UI = DotaUI(self.views[-1])
    self.reward = 0
    self.over_time = self.UI.check_time()
    self.hp = self.UI.get_hp()
    self.gold = self.UI.get_gold()
    self.lvl = self.UI.get_lvl()
    self.ability = self.UI.unlock_ability()

  ## update once the bot makes an action
  def update(self):
    ## screenshot corresponds to the action by the bot
    self.update_views()
    self.UI.update(self.views[-1])
    self.update_reward()
    self.over_time = self.UI.check_time()

  def update_views(self):
    center_hero()
    self.views = [np.array(pg.screenshot())]
    tmp = pg.PAUSE
    pg.PAUSE = 0.1
    self.views.append(np.array(pg.screenshot()))
    pg.PAUSE = tmp

  def update_reward(self):
    UI = self.UI
    hp = UI.get_hp()
    lvl = UI.get_lvl()
    gold = UI.get_gold()
    ability = UI.unlock_ability()
    delta_hp = hp - self.hp
    delta_lvl = lvl - self.lvl
    delta_gold = gold - self.gold
    delta_ability = ability - self.ability
    if delta_gold < 20:
      delta_gold = 0
    ## only considering losing hp
    if delta_hp > 0:
      delta_hp = 0
    self.reward = delta_gold * 2 + delta_lvl * 100 + delta_hp
    self.hp = hp
    self.lvl = lvl
    self.gold = gold
    self.ability = ability

class DotaBot:
  MEMORY_LIMIT = 1000
  MEMORY_RETRIEVAL = 6
  def __init__(self):
    self.env = DotaEnv()
    self.policy = BotPolicy(self)
    self.memory = []
    self.center_x = _width / 2
    self.center_y = _height / 2

  ## interpret the commands and execute them
  def onestep(self):
    ## generate the commands based on the current state
    views = self.env.views
    policy = self.policy
    X = policy.get_state(views[-1], views[-2], self.policy.scale)
    p, meta, prob_mode = policy.forward(X)
    direction, mode = policy.execute(p, prob_mode)
    if len(self.memory) >= self.MEMORY_LIMIT:
      ## randomly throw away old record
      i = np.random.randint(len(self.memory) - self.MEMORY_RETRIEVAL)
      self.memory.pop(i)
    self.memory.append([p.copy(), meta.copy(), direction.copy(), prob_mode, mode])

  def get_parameters(self):
    return self.policy.paras

  def set_parameters(self, parameters):
    paras = self.get_parameters()
    for k in parameters:
      paras[k] = parameters[k]

  def get_UI(self):
    return self.env.UI


class BotPolicy:
  BLACKPIXEL_PERCENT = 0.95
  LEFT_PERCENT = 0.1
  NUM_ACTIONS = 10
  ## add random walk to avoid local minima
  RANDOM_PROB = 0.005
  RANDOM_DIST = 450
  RANDOM_PAUSE = 3
  def __init__(self, bot):
    self.bot = bot
    self.scale = 10 # scaling the screenshot to reduce the dimension
    self.paras = {}
    self.paras['w_fc1'] = np.random.normal(loc=0, scale=0.05, \
      size=[_width // self.scale * _height // self.scale * 2, 100])
    ## output eight direction
    self.paras['w_fc2'] = np.random.normal(loc=0, scale=0.05, \
                                           size=[100, self.NUM_ACTIONS])
    ## the maximum score at the end of the game in the history
    ## the formula is gold + 100 * lvl
    self.paras['max_score'] = 1040
    ## TODO: tune the parameters
    self.battle_pause = 0.3
    self.battle_L = 100 
    self.walk_pause = 1.3
    self.walk_L = 300 
    self.learning_rate = 0.00001
    self.batch_size = 50
    ## the baseline score assuming that the bot did nothing
    self.standard_score = 1040

  ## return the location of the click for a given state
  def forward(self, X):
    ## fully connected layer
    w_fc1 = self.paras['w_fc1']
    X_flatten = X.flatten(order='F')
    X_flatten = np.matrix(X_flatten)
    fc1 = X_flatten.dot(w_fc1)
    ## relu
    fc1[fc1 < 0] = 0
    ## second fully connect layer
    w_fc2 = self.paras['w_fc2']
    fc2 = fc1.dot(w_fc2)
    ## stable softmax
    fc2_p = fc2[0, 0:9]
    fc2_p -= np.max(fc2_p)
    fc2_mode = fc2[0, 9]
    ## probability of taking each direction
    p = np.exp(fc2_p)
    p = p / np.sum(p)
    ## store results for backpropogation
    meta = [X, fc1]
    ## step size
    prob_mode = 1 / (1 + np.exp(-fc2_mode))
    return p, meta, prob_mode

  ## return the gradient of parameters
  def backward(self, p, meta, direction, prob_mode, mode):
    reward = self.bot.env.reward
    X, fc1 = meta
    X_flatten = X.flatten(order='F')
    X_flatten = np.matrix(X_flatten)

    i = direction.argmax()
    dp = np.zeros_like(p)
    for j in range(len(dp)):
      if j == i:
        dp[0, j] = -(1 - p[0, i])
      else:
        dp[0, j] = p[0, j]
    print(fc1.shape)
    print(dp.shape)
    dw_fc2 = fc1.T.dot(dp) 
    w_fc2 = self.paras["w_fc2"]
    dx_fc2 = dp.dot(w_fc2.T[0:9, :])

    ## step_size
    if mode == 1:
      dprob_mode = -(1 - prob_mode)
    else:
      dprob_mode = prob_mode / (self.walk_pause / self.battle_pause)
    print(dw_fc2.shape)
    print((fc1.T.dot(np.matrix(dprob_mode))).shape)
    dw_fc2 = np.concatenate([dw_fc2, fc1.T.dot(np.matrix(dprob_mode))], axis=1)
    print(dw_fc2.shape)
    dx_fc2 += np.matrix(dprob_mode) * w_fc2.T[9, :]
    print(dx_fc2.shape)

    ## relu
    dx_fc2[dx_fc2 < 0] = 0
    ## the first layer
    dw_fc1 = X_flatten.T.dot(dx_fc2)

    return (dw_fc1, dw_fc2)

  def local_optimizer(self):
    reward = self.bot.env.reward
    if reward != 0:
      print(reward)
      dw_fc1 = np.zeros_like(self.paras['w_fc1'])
      dw_fc2 = np.zeros_like(self.paras['w_fc2'])
      l = min(len(self.bot.memory), self.bot.MEMORY_RETRIEVAL)
      for i in range(-1, -(l+1), -1):
        p, meta, direction, prob_mode, mode, _ = self.bot.memory[i]
        x, y= self.backward(p, meta, direction, prob_mode, mode)
        dw_fc1 += x
        dw_fc2 += y
      dw_fc1 /= l
      dw_fc2 /= l

      ## update the parameter
      self.paras['w_fc1'] -= dw_fc1 * self.learning_rate * reward
      self.paras['w_fc2'] -= dw_fc2 * self.learning_rate * reward


  def global_optimizer(self):
    ## increase or desease the amount of gold compared with previous match
    max_score = self.paras['max_score']
    reward = 0
    score = self.bot.env.gold + 100 * self.bot.env.lvl
    if score < self.bot.policy.standard_score:
      reward = min(score - self.bot.policy.standard_score, -100)
    elif score > max_score:
      reward = score - max_score
    self.paras['max_score'] = max(max_score, score)
    ## update the parameter if reward is nonzero
    if reward != 0:
      batch_size = self.batch_size
      if reward > 0:
        ## try to update effective actions
        rewards = [i[3] for i in self.bot.memory]
        pos_indexes = [k for k, v in enumerate(rewards) if v > 0]
        if len(pos_indexes) > 0:
          l = max(pos_indexes)
        else:
          l = len(self.bot.memory)
      else:
        l = len(self.bot.memory)
      ## STG
      batch = (l - 1) // batch_size + 1
      for i in range(batch):
        start = i * batch_size
        end = (i+1) * batch_size
        dw_fc1 = np.zeros_like(self.paras['w_fc1'])
        dw_fc2 = np.zeros_like(self.paras['w_fc2'])
        for j in self.bot.memory[start: end]:
          p, meta, direction, prob_mode, mode, _ = j
          x, y= self.backward(p, meta, direction, prob_mode, mode) 
          dw_fc1 += x
          dw_fc2 += y
        dw_fc1 /= batch_size
        dw_fc2 /= batch_size
        ## update the parameter
        self.paras['w_fc1'] -= dw_fc1 * self.learning_rate * reward 
        self.paras['w_fc2'] -= dw_fc2 * self.learning_rate * reward

  ## negative log likelihood
  def loss(self):
    l = min(len(self.bot.memory), self.bot.MEMORY_RETRIEVAL)
    reward = self.bot.env.reward
    logp = 0
    for i in range(-1, -(l+1), -1):
      p, meta, direction, prob_mode, mode, _ = self.bot.memory[i]
      prob = p.dot(direction)
      logp += np.log(prob) + np.log(mode * prob_mode + (1 - mode) * (1 - prob_mode)) 

    return -logp * reward

  def get_state(self, view1, view2, scale):
    ## use the difference
    X = view1 - view2
    X = np.mean(X[:, :, 0:3], axis=2)
    v = view1.copy()
    v = np.mean(v[:, :, 0:3], axis=2)
    width_per_block, height_per_block = _width // 10, _height // 10

    for i in np.arange(0, _width, width_per_block):
      for j in np.arange(0, _height, height_per_block):
        i = int(i); j = int(j)
        block = X[i:i+width_per_block, j:j+height_per_block]
        ## set entire block to 0 if high percentage of pixels are 0
        if np.sum(block == 0) / (width_per_block * height_per_block) > self.BLACKPIXEL_PERCENT:
          X[i:i+width_per_block, j:j+height_per_block] = 0
    ## reduce the dimension of the input by a factor of scale**2
    X_reduce = np.zeros([_height // scale, _width // scale])
    v_reduce = np.zeros([_height // scale, _width // scale])
    for i in np.arange(0, _height, scale):
      for j in np.arange(0, _width, scale):
        i = int(i); j = int(j)
        X_reduce[i // scale, j // scale] = np.mean(X[i:i+scale, j:j+scale])
        v_reduce[i // scale, j // scale] = np.mean(v[i:i+scale, j:j+scale])
    ## normalize
    X_reduce /= 255
    v_reduce /= 255
    return np.stack([X_reduce, v_reduce], axis=2)

  def execute(self, p, prob_mode):
    center_hero()
    ## TODO: tune the parameter
    tmp = pg.PAUSE
    ## left click happens rare in this case

    # if np.random.binomial(1, self.LEFT_PERCENT) == 1:
    #   button = 'left'
    # else:
    #   button = 'right'
    button = 'right'

    # add randomness to avoid local minimum
    if np.random.binomial(1, self.RANDOM_PROB) == 1:
      print("random walk")
      direction = np.random.multinomial(1, [1.0/self.NUM_ACTIONS]*self.NUM_ACTIONS, size=1)
      pg.PAUSE = self.RANDOM_PAUSE
      L = self.RANDOM_DIST
      mode = 0
    else:
      p = np.squeeze(np.asarray(p))
      direction = np.random.multinomial(1, p)
      mode = np.random.binomial(1, prob_mode)
      if mode == 1:
        L = self.battle_L
        pg.PAUSE = self.battle_pause
      else:
        L = self.walk_L
        pg.PAUSE = self.walk_pause
    i = direction.argmax()
    if i <= 7:
      x = self.bot.center_x + np.cos(i*np.pi / 4) * L
      y = self.bot.center_y + np.sin(i*np.pi / 4) * L
    else:
      x = self.bot.center_x
      y = self.bot.center_y
    pg.click(x=x, y=y, button=button)
    pg.PAUSE = tmp
    print(p)
    print(direction)
    print(prob_mode) 
    print(mode)
    return direction, mode

class DotaUI:
  ## coordinates of key components in Dota 2
  CONTINUE = (956, 904)
  PLAY_DOTA = (1696, 1034)
  CREATE_LOBBY = (1660, 390)
  START_GAME = PLAY_DOTA
  MIRANA = (992, 417)
  BREWMASTER = (411, 226)
  THIRD_ABILITY = (1003, 921)
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
