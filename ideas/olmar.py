

import pandas as pd
import numpy as np
from time import time
from ideas.portfolio import Portfolio as portfolio
import matplotlib.pyplot as plt

class olmar(portfolio):
  def __init__(self, weights = np.nan ):
    self.eps = 10
    self.w = 5
    self.b = weights

  def train(self,train_data):
    X = np.array(train_data)
    (n, m) = X.shape
    self.b = OLMAR_portfolio_sel(self.eps, self.w, X, n)

  def get_portfolio(self, train_data):
    X = np.array(train_data)
    (n, m) = X.shape
    Y = np.ones(X.shape)
    for i in range(1, X.shape[0]):
      Y[i] = X[i] / X[i - 1]
    inv_Y = 1 / Y
    t= n-1
    pred_next_price = (1 + np.sum(np.array([np.prod(inv_Y[ t- i:t + 1, :], axis=0) for i in range(self.w - 1)]),
                                  axis=0)) / self.w
    self.b = OLMAR(self.eps, self.w, pred_next_price, self.b)
def simplex_proj(y):
  m = len(y)
  bget = False

  s = sorted(y, reverse=True)
  tmpsum = 0.0

  for ii in range(m - 1):
      tmpsum = tmpsum + s[ii]
      tmax = (tmpsum - 1) / (ii + 1)
      if tmax >= s[ii + 1]:
          bget = True
          break

  if not bget:
      tmax = (tmpsum + s[m - 1] - 1) / m

  return np.maximum(y - tmax, 0.0)

def new_simplex_proj(x):
  m = x.size

  s = np.sort(x)[::-1]
  c = np.cumsum(s) - 1.
  h = s - c / (np.arange(m) + 1)

  r = np.max(np.where(h>0)[0])
  t = c[r] / (r + 1)

  return np.maximum(0, x - t)

def OLMAR(eps, w, pred_next_price, b_t):
  # calculate average relative predicted price
  m = pred_next_price.size
  avg_pred_price = np.mean(pred_next_price)

  # calculate next Lagrange multiplier
  ones = np.ones(m)
  next_lam = (eps - b_t@pred_next_price) / np.square(np.linalg.norm(pred_next_price - avg_pred_price))
  next_lam = max(0.0, next_lam)

  next_b = b_t + next_lam*(pred_next_price - avg_pred_price)
  normalized_b = new_simplex_proj(next_b)
  # if np.count_nonzero(normalized_b) == 1:
  #   print(next_b)
  return normalized_b

def OLMAR_portfolio_sel(eps, w, X, n):
  # init
  S = 1
  m = X[0].size
  inv_X = 1 / X
  Y = np.ones(X.shape)
  for i in range(1, X.shape[0]):
    Y[i] = X[i] / X[i-1]
  inv_Y = 1/Y
  b = np.ones(m)/m

  Ss = [1]

  for t in range(n):
    # calculate daily return and cumulative return
    S = S * (b @ Y[t])
    # predict next price relative vector
    if t < w-2:
    #   # curr_w = t + 1
    #   # win_range = t+2
      continue
    else:
      win_range, curr_w = w, w
    # window = np.array([np.prod(inv_X[t-i:t+1,:], axis=0) for i in range(win_range-2)])
    pred_next_price = (1 + np.sum(np.array([np.prod(inv_Y[t-i:t+1,:], axis=0) for i in range(win_range-1)]), axis=0)) / curr_w
    b = OLMAR(eps, curr_w, pred_next_price, b)
    Ss.append(S)
  return b

def best_constant_stock(X):
  m = X[0].size
  inv_X = 1 / X
  Y = np.ones(X.shape)
  Ss = [1]
  for i in range(1, X.shape[0]):
    Y[i] = X[i] / X[i-1]
  inv_Y = 1/Y
  best_stock = 0
  max = 0
  S=1
  for col in range(X.shape[1]):
    S=1
    for t in range(X.shape[0]):
      S *= Y[t][col]
    if S>=max:
      best_stock = col
      max = S
  for t in range(X.shape[0]):
    S *= Y[t][best_stock]
    Ss.append(S)
  return best_stock, Ss


def aux_OLMAR_results(data):
  b, Ss = OLMAR_portfolio_sel(eps=10, w=5, X=data, n=data.shape[0])

  xs = np.arange(len(Ss))

  return xs, Ss


def epsilon_checks(X, w=5):
  new_list = range(5, 105)
  eps_to_check = [1, 2]
  for eps in new_list[::5]:
    eps_to_check.append(eps)
  Ss = []
  eps_to_check = np.array(eps_to_check)

  (n, m) = X.shape

  for eps in eps_to_check:
    _, total_wealth = OLMAR_portfolio_sel(eps, w, X, n)
    Ss.append(total_wealth[-1])

  return eps_to_check, Ss

def ws_check(X, eps=10):
  ws_to_check = [3]
  for w in np.arange(5, 105, step=5):
    ws_to_check.append(w)
  # ws_to_check = np.arange(5, 105, step=5)
  ws_to_check = np.array(ws_to_check)
  Ss = []

  (n, m) = X.shape

  for w in ws_to_check:
    _, total_wealth = OLMAR_portfolio_sel(eps, w, X, n)
    Ss.append(total_wealth[-1])

  return ws_to_check, Ss


def plot_w(func, mark=False):
  fig, ax = plt.subplots(2, 2, figsize=(7, 7))
  fig.suptitle("OLMAR Returns compared to baseline")

  # plt.yscale('log')

  for i, (name, title) in enumerate(zip(data_names, data_titles)):
    data = pd.read_csv(f"{name}.csv").to_numpy()

    x, y = func(data)

    ax[i//2, i%2].set_title(title)

    best_stock, z = best_constant_stock(data[3:,:])

    xlabel = 'Epsilon' if mark else 'time'

    ax[i//2, i%2].semilogy(x, y, label='OLMAR')
    ax[i//2, i%2].loglog(x, z[-1]*np.ones(x.size), marker='^')


    ax[i//2, i%2].set_xlabel(xlabel)
    ax[i//2, i%2].set_ylabel("Total Wealth Achieved")
    ax[i//2, i%2].legend(loc="upper left")

  plt.legend()
  fig.tight_layout()
  plt.show()

