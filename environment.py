import numpy as np
import random

class Test_env():
	def __init__(self):
		self.S = np.array([0, 1, 2, 3]) # 状態
		self.A = np.array([0, 1]) # 行動
		self.R = np.array([[0, 1], [-1, 1], [5, -100], [0, 0]]) # 状態n(nは0以上3以下)での行動に対する報酬
		self.S1 = np.array([[2, 1], [0, 3], [3, 0], [None, None]]) # 状態n(nは0以上3以下)での行動した結果の遷移先

		self.epi = 3000
