import numpy as np
import matplotlib.pyplot as plt
import random

from environment import *
from action_value_functions import *

def main():
	env = Test_env()
	# avf = Sarsa(env.R, 0.95)
	avf = Qlerning(env.R, 0.95)
	avf.initialize()

	result_step = []
	result_q = []
	temp = []

	for step in range(env.epi):
		# avf.function(env.S, env.A, env.S1, env.R)
		avf.function(env.S1, env.R)
		print(step)
		if (step + 1) % 10 == 0:
			result_step.append(step + 1)
			result_q.append(avf.Q.tolist())

	result_q = np.array(result_q).transpose([1, 2, 0])

	# s1からのa1とa2の行動価値関数を表示
	s1 = 0
	for a in range(result_q[s1].shape[0]):
		if env.S1[s1][a] == None:
			continue
		else:
			plt.plot(result_step, result_q[s1][a], label='Q(s{}, a{})'.format(s1 + 1, a + 1))

	print(avf.Q)

	plt.legend(loc='lower right')
	plt.show()

main()
