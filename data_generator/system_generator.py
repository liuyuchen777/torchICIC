from const import *
import numpy as np
import matplotlib.pyplot as plt

from channel_generator_3d import single_channel_generator_3d


def place_ut():
    '''
    This function uniformly-randomly place ut
    onto the hexagon area of diameter of area_length.
    Noted that UT and BS aren't allowed to be at the same position.

    Returns:
    ut_pos -- np array in shape of (ut_num, 2)
    ut_dir -- np array in shape of (ut_num,) random direction uiniformly sampled from [0,360)
    '''
    ut_pos = np.empty(shape=(ut_num,2))
    ut_angle = np.random.uniform(0., 2. * np.pi, (ut_num,1))
    ut_dir = np.concatenate((np.cos(ut_angle), np.sin(ut_angle), np.zeros(ut_angle.shape)), -1)
    for i in range(ut_num):
        while 1:
            x = np.random.rand(1) * area_length
            y = np.random.rand(1) * area_length * np.sqrt(3) / 2.
            if y <= -np.sqrt(3) * x + area_length / 4. * np.sqrt(3):
                continue
            elif y >= np.sqrt(3) * x + area_length / 4. * np.sqrt(3):
                continue
            elif y <= np.sqrt(3) * x - area_length / 4. * 3. * np.sqrt(3):
                continue
            elif y >= -np.sqrt(3) * x + area_length / 4. * 5. * np.sqrt(3):
                continue
            else:
                break
        ut_pos[i][0] = x
        ut_pos[i][1] = y
    ut_pos = np.concatenate((ut_pos, ut_height * np.ones(np.shape(ut_angle))), -1)

    '''
    kk = 5
    ut_pos[0] = np.asarray([7.5+kk/2., np.sqrt(3)/2.*kk, 1.5])
    kk = 10
    ut_pos[1] = np.asarray([7.5+kk/2., np.sqrt(3)/2.*kk, 1.5])
    kk = 30
    ut_pos[2] = np.asarray([7.5+kk/2., np.sqrt(3)/2.*kk, 1.5])
    '''
    
    return ut_pos, ut_dir
    
def cal_dis(bs_pos, ut_pos):
    '''
    This function calculate distance between every possible [BS,UT] pair.
    
    Arguments:
    bs_pos -- np array in shape of (bs_num, 2)
    ut_pos -- np array in shape of (ut_num, 2)

    Returns:
    dis -- np array in shape of (bs_num, ut_num), distance between each [BS,UT] pair
    '''
    dis = np.empty(shape=(bs_num,ut_num))
    
    for b in range(bs_num):
        dis[b, :] = np.sqrt(np.sum((bs_pos[b, :] - ut_pos)**2, axis=1))
    
    return dis
    

def cal_AoD_AoA(bs_pos, bs_dir, ut_pos, ut_dir):
    '''
    This function calculate distance between every possible [BS,UT] pair.
    
    Arguments:
    bs_pos -- np array in shape of (bs_num, 2)
    bs_dir -- np array in shape of (bs_num,) random direction uiniformly sampled from [0,360)
    ut_pos -- np array in shape of (ut_num, 2)
    ut_dir -- np array in shape of (ut_num,) random direction uiniformly sampled from [0,360)

    Returns:
    AoD -- np array in shape of (bs_num, ut_num), AoD of each [BS,UT] pair
    AoA -- np array in shape of (bs_num, ut_num), AoA of each [BS,UT] pair
    '''
    def get_angle(x, y):
        r = np.arccos(x / np.sqrt(x**2 + y**2))
        if y < 0:
            r = -r
        return r
        

    AoD = np.empty(shape=(bs_num,ut_num,2))
    AoA = np.empty(shape=(bs_num,ut_num))

    for b in range(bs_num):
        for u in range(ut_num):
            delta = ut_pos[u] - bs_pos[b]
            delta /= np.linalg.norm(delta)
            theta = np.arccos(np.dot(delta, bs_dir[b, 2]))
            phi = get_angle(np.dot(delta, bs_dir[b, 0]), np.dot(delta, bs_dir[b, 1]))
            AoD[b, u, 0] = theta
            AoD[b, u, 1] = phi

            delta = bs_pos[b, :-1] - ut_pos[u, :-1]
            delta /= np.linalg.norm(delta)
            tmp1 = get_angle(delta[0], delta[1])
            tmp2 = get_angle(ut_dir[u, 0], ut_dir[u, 0])

            AoA[b, u] = (tmp1 - tmp2)
    return AoD, AoA

def cal_path_loss(dis):
    '''
    This function calculate path loss between every possible [BS,UT] pair.

    Arguments:
    dis -- np array in shape of (bs_num, ut_num), distance between each [BS,UT] pair

    Returns:
    path_loss -- np array in shape of (bs_num, ut_num), path_loss between each [BS,UT] pair
    '''
    path_loss = np.empty(shape=(bs_num,ut_num)) 
    for b in range(bs_num):
        for u in range(ut_num):
            '''
            ita = 10**(log_std / 10) * np.random.randn()
            ita = np.exp(ita)
            '''

            ita = log_std * np.random.randn()
            ita =  10**(ita / 10)

            path_loss[b][u] = dis[b][u]**path_loss_exponent * ita
    return path_loss

def assign(H, path_loss, ut_pos):
    '''
    This function groups UTs and BSs into pairs.
    Arguments:
    H -- np array, channel matrixs without pathloss of the system at a single time instant.
         in shape of (bs_num, ut_num, delay, n_r, n_t)
    path_loss -- np array in shape of (bs_num, ut_num), path_loss between each [BS,UT] pair
    ut_pos -- np array in shape of (ut_num, 2), the sequence of ut_pos will be shuffled by
              this function to match with G

    Returns:
    G -- np array, channel matrixs with pathloss of the system at a single time instant.
         in shape of (bs_num, ut_num, n_r, n_t). The sequence of G has been shuffled so that
         i-th UT is linked with i-th BS. Channels of different path(delay) are accumulated.
    G_power -- np array, channel power(pathloss included) in shape of (bs_num, ut_num).
         The sequence of G_power has been shuffled so that i-th UT is linked with i-th BS.
    '''
    # For each antanna pair, add channels of all paths together
    H = np.sum(H, axis=2)
    # Expand dims of path_loss to be compatible with G
    path_loss_expand = np.expand_dims(path_loss, axis=-1)
    path_loss_expand = np.expand_dims(path_loss_expand, axis=-1)
    G =  np.sqrt(1 / path_loss_expand) * H
    
    # G_power = np.real(np.sum(G * np.conjugate(G), axis=(-1, -2)))
    G_power = np.linalg.norm(G, axis=(-2,-1),ord=2) #, ord=2
    valid = np.ones(shape=path_loss.shape)
    pair_temp = []
    for _ in range(link_num):
        value_temp = 0
        index_temp = []
        for b in range(bs_num):
            for u in range(ut_num):
                if valid[b][u] and G_power[b, u] > value_temp:
                    value_temp = G_power[b, u]
                    index_temp = [b, u]
        pair_temp.append(index_temp)
        for b in range(bs_num):
            valid[b, index_temp[1]] = 0
        for u in range(ut_num):
            valid[index_temp[0], u] = 0
    pair_temp.sort(key = lambda a: a[0])

    pair = [i[1] for i in pair_temp]

    G = G[:, pair, :, :]
    G_power = G_power[:, pair]
    ut_pos = ut_pos[pair, :]
    path_loss = path_loss[:, pair]

    return G, G_power

def plot_system_map(bs_pos, ut_pos):
    '''
    This function plots a system map according to BS and UT positons.
    Links are connected by lines

    Arguments:
        bs_pos -- np array in shape of (bs_num, 2)
        ut_pos -- np array in shape of (ut_num, 2)
    '''

    plt.xlim(0, area_length)
    plt.ylim(0, area_length)
    
    for b in range(bs_num):
        plt.plot(bs_pos[b, 0],bs_pos[b, 1],'o',color='darkred')
        plt.text(bs_pos[b, 0] - 0.1 ,bs_pos[b, 1] + 0.3 ,str(b), color='darkred')
    for u in range(ut_num):
        plt.plot(ut_pos[u, 0],ut_pos[u, 1],'x', color='darkblue')
        plt.text(ut_pos[u, 0] - 0.1 ,ut_pos[u, 1] + 0.3 , str(u), color='darkblue')

    for i in range(link_num):
        x = [bs_pos[i, 0], ut_pos[i, 0]]
        y = [bs_pos[i, 1], ut_pos[i, 1]]
        plt.plot(x, y, color='darkgreen')
    plt.grid()
    plt.gca().set_aspect(1)
    plt.show()

    
def system_generator(ifplot = True):
    '''
    This function generates a communication system.
    
    Argument:
    ifplot -- boolean, if True, plot system map.
    Returns:
    G -- np array, channel matrixs with pathloss of the system at a single time instant.
         in shape of (bs_num, ut_num, n_r, n_t). The sequence of G has been shuffled so that
         i-th UT is linked with i-th BS. Channels of different path(delay) are accumulated.
    G_power -- np array, channel power(pathloss included) in shape of (bs_num, ut_num).
         The sequence of G_power has been shuffled so that i-th UT is linked with i-th BS.
    '''
    # generate area map

    ut_pos, ut_dir = place_ut()

    # generate random path loss
    dis = cal_dis(bs_pos, ut_pos)
    path_loss = cal_path_loss(dis)
    AoD, AoA = cal_AoD_AoA(bs_pos, bs_dir, ut_pos, ut_dir)

    # generate random channels
    H = np.zeros((bs_num,ut_num,max_delay,n_r,n_t), dtype=complex)
    for b in range(bs_num):
        for u in range(ut_num):
            H[b,u,:,:,:], _, _ = single_channel_generator_3d(K_factor, N_path, N_component, max_delay, AoD[b, u], AoA[b, u])
    
    
    '''
    G = np.sum(H, axis=2)
    GF = []
    for j, F in enumerate(precoding_matrices):
        GF.append(G@F)
    GF = np.asarray(GF)
    GF_power = np.sum(np.abs(GF)**2, axis=(-1,-2))

    print(np.where(GF_power[:,0,0]==np.max(GF_power[:,0,0])))
    print(np.where(GF_power[:,0,1]==np.max(GF_power[:,0,1])))
    print(np.where(GF_power[:,0,2]==np.max(GF_power[:,0,2])))
    '''
    
    
    G, G_power = assign(H, path_loss, ut_pos)
    if ifplot:
        plot_system_map(bs_pos,ut_pos)

    return G, G_power


if __name__ == '__main__':
    for i in range(1):
        if (i+1)%1000 == 0:
            print(i)   
        G, G_power = system_generator(True)
