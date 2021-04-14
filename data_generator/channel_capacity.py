from const import n_t, n_r, bs_num, ut_num, n_stream, NOISE_POW, check_nan, CQI_to_throughput
import numpy as np

def channel_capacity_per_user(G, u, pair, signal_pow, precoding_index):
	'''
	This function calculate channel capacity for a user
		
	Argument:
	R -- np array(double), channel corelation matrices of the system at a single time instant, pathloss already included.
		 in shape of (bs_num, ut_num, len(precoding_matrices),n_r, n_stream)
	u -- int, user index
	pair -- python list, indicates the BS-UT pairs.
	        Each element is a python list of user index assigned to that BS.
	signal_pow -- python list, indicates transmitting power of each BS in dBm.
	Returns:
	C -- float, channel capacity for user u
	'''
	
	interference_power = 0.
	bs = u
	for b in range(bs_num):
		if b != bs:
			temp = 1 / 1000 * np.power(10, signal_pow[b] / 10) / n_t / n_stream * abs(G[bs, u, precoding_index[bs]].T.conj()@G[b, u, precoding_index[b]])**2
			if check_nan(temp):
				print('nan')
			interference_power += temp
	interference_power += NOISE_POW * np.sum(np.abs(G[bs, u, precoding_index[bs]])**2) # np.abs(np.sum(G[bs, u, precoding_index[bs]]))**2
	
	desired_power = 1 / 1000 * np.power(10, signal_pow[bs] / 10) / n_t / n_stream * abs(G[bs, u, precoding_index[bs]].T.conj()@G[bs, u, precoding_index[bs]])**2

	SINR = desired_power / interference_power
	C = np.log2(np.linalg.det(1 + SINR))


	if check_nan(C):
		print('nan')
	return C.real

def system_capacity(G, pair, signal_pow, precoding_index):
	'''
	This function calculate system capacity

	Argument:
	R -- np array(double), channel corelation matrices of the system at a single time instant, pathloss already included.
		 in shape of (bs_num, ut_num, len(precoding_matrices),n_r, n_stream)
	u -- int, user index
	pair -- python list, indicates the BS-UT pairs.
	        Each element is a python list of user index assigned to that BS.
	signal_pow -- python list, indicates transmitting power of each BS in dBm.
	Returns:
	C -- float, system average capacity
	'''
	C = 0
	for u in range(ut_num):
		C += channel_capacity_per_user(G, u, pair, signal_pow, precoding_index)
	return C / len(pair)

def channel_throughput_per_user(G, u, pair, signal_pow, precoding_index):
	'''
	This function calculate channel throughput for a user
		
	Argument:
	R -- np array(double), channel corelation matrices of the system at a single time instant, pathloss already included.
		 in shape of (bs_num, ut_num, len(precoding_matrices),n_r, n_stream)
	u -- int, user index
	pair -- python list, indicates the BS-UT pairs.
	        Each element is a python list of user index assigned to that BS.
	signal_pow -- python list, indicates transmitting power of each BS in dBm.
	Returns:
	throughput -- float, channel throughput for user u
	CQI -- int, CQI of this user under the given power configuration
	'''

	interference_power = 0.
	bs = u
	for b in range(bs_num):
		if b != bs:
			temp = 1 / 1000 * np.power(10, signal_pow[b] / 10) / n_t / n_stream * abs(G[bs, u, precoding_index[bs]].T.conj()@G[b, u, precoding_index[b]])**2
			if check_nan(temp):
				print('nan')
			interference_power += temp
	interference_power += NOISE_POW * np.abs(np.sum(G[bs, u, precoding_index[bs]]))**2
	
	desired_power = 1 / 1000 * np.power(10, signal_pow[bs] / 10) / n_t / n_stream * abs(G[bs, u, precoding_index[bs]].T.conj()@G[bs, u, precoding_index[bs]])**2

	SINR = desired_power / interference_power	
	
	if SINR<=0:
		print('!')
		print('SINR = ' + str(SINR))
		print('desired_signal_power = ' + str(desired_power))
		print('interference_power = ' + str(interference_power))
	
		print('u = ' +str(u))
		print('channel power = '+ str(np.real(np.sum(R[u, u, precoding_index[u]]))))
		print('power gain = ' + str(1 / 1000 * np.power(10, signal_pow[u] / 10) / n_t))
		print('channel')
		print(G[u, u, precoding_index[u]])
		np.save('bad_data.npy', G[u, u, precoding_index[u]])
		
	SNR_eff_dB = 10 * np.log10(SINR) if SINR > 0 else -10
	CQI = int(np.floor(0.5223 * SNR_eff_dB + 4.6176))
	CQI = min(CQI, 15)
	CQI = max(CQI, 0)
	throughput = CQI_to_throughput[CQI]
	return throughput, CQI

def system_throughput(G, pair, signal_pow, precoding_index):
	'''
	This function calculate system throughput

	Argument:
	R -- np array(double), channel corelation matrices of the system at a single time instant, pathloss already included.
		 in shape of (bs_num, ut_num, len(precoding_matrices),n_r, n_stream)
	u -- int, user index
	pair -- python list, indicates the BS-UT pairs.
	        Each element is a python list of user index assigned to that BS.
	signal_pow -- python list, indicates transmitting power of each BS in dBm.
	precoding_index -- python list, indicates precoding matrix index of each BS.
	Returns:
	throughput -- float, system average throughput
	CQI -- python list, each element is a int indicating CQI to the corresponding link
	'''
	throughput = 0
	CQI = []
	for u in range(ut_num):
		throughput_temp, CQI_temp = channel_throughput_per_user(G, u, pair, signal_pow, precoding_index)
		throughput += throughput_temp
		CQI.append(CQI_temp)
	throughput /= len(pair)
	return throughput, CQI

