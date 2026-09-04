import numpy as np

def boolean_transform(x, a):
  """
  一般化ブール変換
  input: 
    x: 入力値
    a: 変換パラメータ
  
  return:
    ブール変換の結果
  """
  e = 1e-10
  return a * (x + 1/(x + e))


