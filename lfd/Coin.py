import numpy as np
import random as rd

def focus_on(array, position=0):
	return array[position]

def toss(num_of_coins=1000, num_of_tosses=10):
	E_prob = np.zeros(num_of_coins)
	for coin in range(num_of_coins):
		expected_head = 0
		for toss in range(num_of_tosses):
			expected_head = expected_head + (rd.randint(0,1)*(1.0/num_of_tosses))
		E_prob[coin] = expected_head
	return E_prob

def main():
	num_of_coins  = 1000
	num_of_tosses = 10
	num_of_experiment = 100000

	v_1_distribution = np.zeros(num_of_experiment) #v_1_distribution
	v_r_distribution = np.zeros(num_of_experiment) #v_rand_distribution
	v_m_distribution = np.zeros(num_of_experiment) #v_min_distribution

	for i in range(num_of_experiment):
		print("TOSS: ", i)
		E_prob = toss(num_of_coins, num_of_tosses)
		v_1_distribution[i] = focus_on(E_prob, 0)
		v_r_distribution[i] = focus_on(E_prob, rd.randint(0, num_of_coins-1))
		v_m_distribution[i] = np.min(E_prob)
		#print v_1, v_rand, v_min
	return v_1_distribution, v_r_distribution, v_m_distribution


if __name__ == '__main__':
	(v_1_distribution, v_r_distribution, v_m_distribution) = main()
	mean_v_1   = np.mean(v_1_distribution)
	mean_v_rand= np.mean(v_r_distribution)
	mean_v_min = np.mean(v_m_distribution)
	print mean_v_1, mean_v_rand, mean_v_min