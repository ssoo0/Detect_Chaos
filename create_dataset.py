import numpy as np
import random
import math
import os
from sklearn.model_selection import train_test_split
import nolds

dirpath = './dataset/'

def logistic_time_series(mu):  # mu=0~3.54409 だと非Chaos,  mu=3.54409~4だとChaos
  n = 1000  # 時刻nまでの時系列を生成
  x = 0.5  # 初期値
  time_sereis = [x]
  for i in range(n - 1):
    x_new = mu * x * (1.0 - x)
    time_sereis.append(x_new)
    x = x_new
    if math.isnan(x) or math.isinf(x):
      ValueError('The value is incorrect!')
  time_sereis = np.array(time_sereis)
  return time_sereis

def sine_circle_time_series(mu):
  n = 1000  # 時刻nまでの時系列を生成
  omega = 0.606661
  theta = 0.5  # 初期値
  time_sereis = [theta]

  for i in range(n - 1):
    theta_new = theta + omega - mu * math.sin(2.0 * math.pi * theta) / (2.0 * math.pi)
    theta_new = theta_new % 1
    time_sereis.append(theta_new)
    theta = theta_new
    if math.isnan(theta) or math.isinf(theta):
      ValueError('The value is incorrect!')
  time_sereis = np.array(time_sereis)
  return time_sereis  

def decide_chaos(time_sereis, threshold_entropy):
  _time_sereis = np.array(time_sereis)
  try:
    if (nolds.lyap_r(_time_sereis) > 0.0 and nolds.sampen(_time_sereis) > threshold_entropy):
      return 1  # 時系列がChaosのときは1を返す
    else:
      return 0  # 時系列が非chaosのときは0を返す
  except:
    return 0


def create_logistic_map_dataset():
  mu_initial = 0.0
  mu_threshold = 3.5699456
  mu_final = 4.0
  number_mu = 32*300 # number_mu個の様々な値のパラメータmuを生成.
  number_chaos = int(number_mu / 2)
  number_nonchaos = number_mu - number_chaos
  all_datasets = []

  # chaosのデータセットを作る
  threshold_entropy = 0.10
  count = 0
  while count < number_chaos:
    mu = random.uniform(mu_threshold, mu_final)
    time_series = logistic_time_series(mu)
    if decide_chaos(time_series, threshold_entropy) == 1:
      dataset = [time_series, 1]
      all_datasets.append(dataset)
      count += 1
  # 非chaosのデータセットを作る
  count = 0
  while count < number_nonchaos:
    mu = random.uniform(mu_initial, mu_final)
    time_series = logistic_time_series(mu)
    if decide_chaos(time_series, threshold_entropy) == 0:
      dataset = [time_series, 0]
      all_datasets.append(dataset)
      count += 1

  data_train, data_test = train_test_split(all_datasets, train_size=6.0/10.0)
  data_val, data_test = train_test_split(data_test, train_size=0.5)

  spilited_datasets = [data_train, data_val, data_test]
  input_set = []
  output_set = []
  x_train, x_val, x_test = [], [], []
  y_train, y_val, y_test = [], [], []
  for dataset in spilited_datasets:
    inputs = []
    outputs = []
    for data in dataset:
      inputs.append(data[0])
      outputs.append(data[1])
    input_set.append(np.array(inputs))
    output_set.append(np.array(outputs))

  x_train, x_val, x_test = input_set
  y_train, y_val, y_test = output_set
  x_train = x_train.reshape(x_train.shape + (1,1,))
  x_val = x_val.reshape(x_val.shape + (1,1,))
  x_test = x_test.reshape(x_test.shape + (1,1,))
  np.save(dirpath + 'x_train', x_train)
  np.save(dirpath + 'x_val', x_val)
  np.save(dirpath + 'x_test', x_test)
  np.save(dirpath + 'y_train', y_train)
  np.save(dirpath + 'y_test', y_test)
  np.save(dirpath + 'y_val', y_val)


def create_logistic_map_dataset_with_noise():
  mu_initial = 0.0
  mu_threshold = 3.5699456
  mu_final = 4.0
  number_mu = 32*300  # number_mu個の様々な値のmuを生成.
  number_chaos = int(number_mu / 2)
  number_nonchaos = number_mu - number_chaos
  all_datasets = []
  noise_strength = 0.01

  # chaosのデータセットを作る
  threshold_entropy = 0.10
  count = 0
  while count < number_chaos:
    mu = random.uniform(mu_threshold, mu_final)
    time_series = logistic_time_series(mu)  + np.random.rand(1000) * noise_strength
    if decide_chaos(time_series, threshold_entropy) == 1:
      dataset = [time_series, 1]
      all_datasets.append(dataset)
      count += 1
  # 非chaosのデータセットを作る
  count = 0
  while count < number_nonchaos:
    mu = random.uniform(mu_initial, mu_final)
    time_series = logistic_time_series(mu) + np.random.rand(1000) * noise_strength
    if decide_chaos(time_series, threshold_entropy) == 0:
      dataset = [time_series, 0]
      all_datasets.append(dataset)
      count += 1

  data_train, data_test = train_test_split(all_datasets, test_size=6.0/10.0)
  data_val, data_test = train_test_split(data_test, test_size=0.5)

  spilited_datasets = [data_train, data_val, data_test]
  input_set = []
  output_set = []
  x_train, x_val, x_test = [], [], []
  y_train, y_val, y_test = [], [], []
  for dataset in spilited_datasets:
    inputs = []
    outputs = []
    for data in dataset:
      inputs.append(data[0])
      outputs.append(data[1])
    input_set.append(np.array(inputs))
    output_set.append(np.array(outputs))

  x_train, x_val, x_test = input_set
  y_train, y_val, y_test = output_set
  x_train = x_train.reshape(x_train.shape + (1,1,))
  x_val = x_val.reshape(x_val.shape + (1,1,))
  x_test = x_test.reshape(x_test.shape + (1,1,))
  np.save(dirpath + 'noise_x_train', x_train)
  np.save(dirpath + 'noise_x_val', x_val)
  np.save(dirpath + 'noise_x_test', x_test)
  np.save(dirpath + 'noise_y_train', y_train)
  np.save(dirpath + 'noise_y_val', y_val)
  np.save(dirpath + 'noise_y_test', y_test)


def create_sine_circle_map_dataset():
  mu_initial = 0.0
  mu_final = 5.0
  number_mu = 1728 # number_mu個の様々な値のmuを生成.
  number_chaos = int(number_mu / 2)
  number_nonchaos = number_mu - number_chaos
  sine_circle_x_test = []
  sine_circle_y_test = []

  # カオスのデータセットを作る
  threshold_entropy = 0.60
  count = 0
  while count < number_chaos:
    mu = random.uniform(mu_initial, mu_final)
    time_series = sine_circle_time_series(mu)  
    if decide_chaos(time_series, threshold_entropy) == 1:
      sine_circle_x_test.append(np.array(time_series))
      sine_circle_y_test.append(np.array(1))
      count += 1
  # 非chaosのデータセットを作る
  count = 0
  while count < number_nonchaos:
    mu = random.uniform(mu_initial, mu_final)
    time_series = sine_circle_time_series(mu)  
    if decide_chaos(time_series, threshold_entropy) == 0:
      sine_circle_x_test.append(np.array(time_series))
      sine_circle_y_test.append(np.array(0))
      count += 1

  sine_circle_x_test = np.array(sine_circle_x_test)
  sine_circle_y_test = np.array(sine_circle_y_test)
  sine_circle_x_test = sine_circle_x_test.reshape(sine_circle_x_test.shape + (1,1,))
  np.save(dirpath + 'sine_circle_x_test', sine_circle_x_test)
  np.save(dirpath + 'sine_circle_y_test', sine_circle_y_test)


if not os.path.exists(dirpath):
  os.mkdir(dirpath)

create_logistic_map_dataset()
create_logistic_map_dataset_with_noise()
create_sine_circle_map_dataset()
