import numpy as np
import random

class Sarsa():
	def __init__(self, R, per):
		self.alpha = 0.01
		self.gamma = 0.8
		self.q_init = 10
		self.Q = np.zeros(R.shape)
		self.per = per

	def initialize(self):
		self.Q[:3] += self.q_init

	def pi(self):
		if random.random() <= self.per:
			return 0
		else:
			return 1

	def function(self, S1, R):
		s = 0
		a = self.pi()
		while(S1[s][a] != None):
			a_next = self.pi()
			next_s = S1[s][a]
			TD = R[s][a] + self.gamma * self.Q[next_s, a_next] - self.Q[s][a]
			self.Q[s][a] += self.alpha * TD
			s = next_s
			a = a_next

class Qlearning():
	def __init__(self, R, per):
		self.alpha = 0.01
		self.gamma = 0.8
		self.q_init = 10
		self.Q = np.zeros(R.shape)
		self.per = per

	def initialize(self):
		self.Q[:3] += self.q_init

	def pi(self):
		if random.random() <= self.per:
			return 0
		else:
			return 1

	def greedy(self, s):
		if self.Q[s][0] > self.Q[s][1]:
			return 0
		else:
			return 1

	def function(self, S1, R):
		s = 0
		a = self.pi()
		while (S1[s][a] != None):
			next_s = S1[s][a]
			TD = R[s][a] + self.gamma * max(self.Q[next_s]) - self.Q[s][a]
			self.Q[s][a] += (self.alpha * TD)
			# print(R[s][a])
			s = S1[s][a]
			a = self.pi()
