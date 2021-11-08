import numpy as np
from ..const import check_nan, bs_num, ut_num, link_num, n_t, n_t_y, n_t_z, n_r, n_interval, precoding_matrices


def Laplace_rand(mu, phi, m, n):
    """
    This function generates an np matrix in shape of (m, n)
    All elements in the matrix are iid Laplace random variables
    with mean = mu and variance = 2*phi^2
    Argument:
        mu -- float, mean of the Laplace random variables
        phi -- parameter of the Laplace random variables, varriance = 2*phi^2
        m -- #row of Laplace_mat
        n -- #col of Laplace_mat
    Returns:
        Laplace_mat -- np matrix in (m, n), all elements in the matrix are iid Laplace random variable
    """
    U_1 = np.random.rand(m, n) # uniform distribution in [0,1)
    U_2 = 2 * np.random.randint(0, 2, size=(m, n)) - 1 # generate -1 and 1 with probability 0.5 respectively
    Laplace_mat = mu - np.log(U_1) * phi * U_2
    return Laplace_mat


def exp_rand(phi, m, n):
    """
    This function generates an np matrix in shape of (m, n)
    All elements in the matrix are iid random variables of exponetial distribution
    Argument:
        phi -- parameter of the exponetial distribution, varriance = 2*phi^2
        m -- #row of exprand_mat
        n -- #col of exprand_mat
    Returns:
        exprand_mat -- np matrix in (m, n), all elements in the matrix are iid random variables of exponetial distribution
    """
    U = np.random.rand(m, n) # uniform distribution in [0,1)
    exprand_mat = -np.log(U) * phi
    return exprand_mat


def single_channel_generator_3d(K_factor, N_path, N_component, max_delay, AoD, AoA):
    """
    This function generates a dictionary contains channel information
    Delays starts from 1 instead of 0, all elements of Delay_LOS equal to 1
    x is channel gain matrix
    Phi is propagation angle of signal
    LOS path only has one components, each NLOS path has multiple components
    Delay of the first component of each path form an ascending array
    Delays of different components of each path form an ascending array
    For a path, Phi of different components are iid Laplace varriable with the same mean, Phi of the first component is mean

    Argument:
        K_factor -- int, Rician factor in dB
        N_path -- int, #path including the LOS path
        N_component -- int, #component of each NLOS path
        max_delay -- int, the max delay of system (GI_point)
        AoD -- float, AoD of BS in rad
        AoA -- float, AoA of BS in rad
    Returns:
        single_3d_channel -- dictionary
            "Delay" = Delay
            "x" = x
            "Phi" = Phi
    """

    Delay = np.zeros(shape=(N_component, N_path), dtype=int)
    x = np.zeros(shape=(N_component, N_path), dtype=complex)

    K = 10**(K_factor / 10)  # Rician factor

    Delay[0, 0] = 1
    while True:
        x_LOS = np.random.randn(1)+np.random.randn(1)*1j
        power_temp = np.abs(x_LOS)**2
        if power_temp > 1e-12: # prevent power_temp == 0.
            break
    x_LOS = (K / (K + 1))**0.5 * x_LOS / (power_temp**0.5)
    x[0,0] = x_LOS

    for I in range(1, N_path):
        Delay_temp = np.ceil(exp_rand(17.60, 1, 1)) + Delay[0,I-1] # Lambda
        if Delay_temp > max_delay:
            break
        Delay[0,I] = Delay_temp

        x_temp = np.exp(-(Delay[0,I] - 1) / 24.55) # Gamma
        x[0,I] = x_temp * (np.random.randn(1) + np.random.randn(1)*1j) / 2**0.5
        
        for J in range(1, N_component):
            Delay_temp = np.round(exp_rand(6.77, 1, 1)) + Delay[J-1,I] # Lambda
            if Delay_temp > max_delay:
                break
            Delay[J, I] = Delay_temp
            # Higher delay leads to fewer channel power
            x[J,I] = x_temp * np.exp(-(Delay[J,I] - Delay[0,I]) / 19.80) * (np.random.randn(1) + np.random.randn(1)*1j) / 2**0.5 # Gamma
    
    IR_NLOS_temp = np.zeros(shape=(max_delay,1), dtype=complex) # Impulse Response
    for I in range(1, N_path):
        for J in range(N_component):
            if Delay[J, I] == 0:
                break
            IR_NLOS_temp[Delay[J, I]-1] += x[J, I]
    power_temp = np.sum(np.abs(IR_NLOS_temp)**2)
    
    if power_temp < 1e-12:
        x[:, 1:] = 0 * x[:, 1:]  # prevent power_temp == 0.
    else:
        x[:, 1:] = (1 / (K + 1))**0.5 * x[:, 1:] / (power_temp**0.5)
 
    # Delay_mean = np.sum(Delay * np.abs(x)**2) / np.sum(np.abs(x)**2)
    # Delay_spread = (np.sum((Delay - Delay_mean)**2 * np.abs(x)**2) / np.sum(np.abs(x)**2))**0.5

    # if check_nan(Delay_spread):
    #     print('NaN occurs in Delay_spread')
    
    Angle_scaler = 0.57
    # AoD and AoA of each component is generated
    Phi = np.zeros(shape=(N_component, N_path, 2))
    Theta = np.zeros(shape=(N_component, N_path))
    Phi[0, 0, 0] = AoD[1]
    Phi[0, 0, 1] = AoA
    Theta[0, 0] = AoD[0]
    for I in range(1, N_path):
        if Delay[0,I] == 0:
            break
        Phi[0, I, 0] = 2 * np.pi * np.random.rand(1)
        Phi[0, I, 1] = 2 * np.pi * np.random.rand(1)
        Theta[0, I] = 2 * np.pi * np.random.rand(1)
        for J in range(1, N_component):
            if Delay[J, I] == 0:
                break
            Phi[J, I, 0] = Laplace_rand(Phi[0, I, 0], Angle_scaler, 1, 1)
            Phi[J, I, 1] = Laplace_rand(Phi[0, I, 1], Angle_scaler, 1, 1)
            Theta[J, I] = Laplace_rand(Theta[0, I], Angle_scaler, 1, 1)

    # Phi_mean = np.sum(Phi * np.abs(x)**2) / np.sum(np.abs(x)**2)
    # Phi_spread = (np.sum((Phi - Phi_mean)**2 * np.abs(x)**2) / np.sum(np.abs(x)**2))**0.5

    H = np.zeros(shape=(N_component,N_path,n_r,n_t), dtype=complex)
    IR = np.zeros(shape=(max_delay,n_r,n_t), dtype=complex)
    for I in range(N_path):
        if Delay[0, I] == 0:
            break
        for J in range(N_component):
            if Delay[J, I] == 0:
                break
            # Array factor of transmitting antennas
            AF_t = np.exp(1j * np.arange(n_t_z) * 2 * np.pi * n_interval * np.cos(Theta[J, I])).reshape((n_t_z,1)) \
                * np.exp(1j * np.arange(n_t_y) * 2 * np.pi * n_interval * np.sin(Phi[J, I, 0]) * np.sin(Theta[J, I])).reshape((1,n_t_y))
            AF_t = AF_t.reshape((1,-1))
            # Array factor of recieving antennas
            #AF_r = np.ones(n_r).reshape((n_r,1))
            AF_r = np.exp(1j * np.arange(n_r) * 2 * np.pi * n_interval * np.sin(Phi[J, I, 1])).reshape((n_r,1))
            #Chnnel on each component and path
            H[J,I,:,:] = np.dot(AF_r, AF_t) * x[J,I]
            #Impulse response on each time tap
            IR[Delay[J,I]-1,:,:] += H[J,I,:,:]
    return IR, H, Delay


if __name__ == '__main__':
    cnt0 = 0
    cnt1 = 0
    for i in range(1000):
        IR, H, Delay = single_channel_generator_3d(10, 6, 8, 32, [30, -60], 60)
        G = np.sum(IR, axis=0)

        GF = np.dot(G, precoding_matrices[2])
        teemp = np.matmul(GF, np.swapaxes(GF,-1,-2).conj())   

        if np.where(teemp!=0)[0].shape[0]==0:
            cnt0 += 1
        else:
            cnt1 += 1

