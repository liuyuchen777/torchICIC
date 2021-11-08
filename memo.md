# Memo

## Code Reading Memo

### Joint Transmit Power and 3D Beamforming Control using Neural Networks for MIMO Small Cell Systems
这篇文章先通过ES计算出received power levels of all the possible equivalent channels对应的最佳power level和beamforming vector
然后以交叉熵为损失函数训练神经网络
所以神经网络中就没有任何与通信公式相关的计算，直接调用之前做好的数据

目前所有生成数据的文件放在data_generator中
1.  test_model.py 测试训练好的数据的准确性
2.  data_generator.py 生成数据的入口文件
    func train_data_generator(num_data, mode)
    [mode] "capacity", "throughput"
    func greedy(G, pair, mode)
    pair = [[0],[1],[2]]
    G_all = all channel of the test data (channel information)
    [mode]
    "capacity" -> func greedy_capacity(G, pair)
    "throughput" -> func greedy_throughput(G, pair)
3.  func greedy_capacity(...), func greedy_throughput(...)依赖channel_capacity.py
    func system_capacity(...) -> func channel_capacity_per_user(...)
    func system_throughput(...) -> func channel_thtoughput_per_user(...)
4.  func test_data_generator(...) -> func system_generator(...)
    生成信道信息？
5.  func system_generator(...) -> func single_channel_generator_3d(...)
    channel_generator_3d.py

通信信道相关的函数全部在data_generator文件夹中，不需要对其进行改动，DQN的代码中也是直接对其进行引用的

### Deep Reinforcement Learning for Distributed Dynamic MISO Downlink-Beamforming Coordination (DDBC)



## TODO

- [x] General Utils
- [x] Rayleigh Channel
- [x] Reward Function Basic
- [x] Interference Penalty in Reward Function
- [x] Skip Sector
- [x] Rename functions and variables
- [x] Build State
- [x] DQN
- [x] Decision Maker
- [x] Check Copy Occur

## Research Memo

2021.4.13
start to rebuild joint transmission power and beamforming vector to pytorch
train_model001-2_best_mod1.py中定义了两种loss，一种MSE，一种cross entropy，最后只用了cross entropy
正则化不知道在PyTorch中怎么处理，就先不管了
pytorch正则化已处理

2021.4.18
论文中神经网络的维度和代码中维度不同
论文:

1) 不加入enhanced data
```
K^2*|F| = 9 * 8
```
2) 加入enhanced data
```
K^2*|F|^2 = 9 * 64
代码中:
17600 * 80
why 80?
```

2021.4.23

NN出来结果非常不理想，基本跟乱猜一样

2021.4.24

之前用的网络只有一个隐藏层，现在尝试4个隐藏层

开始训练时候的cross entropy非常大

蒋学长代码中的power_loss一开始是1.多，beam_loss是2.多，和我的差得有点远

明明是一个公式算出来的

可能原因1：网络参数初始化，蒋学长w初始化为uniform，b初始化0

可能原因2：正则化，我用的是Adam优化器自带的正则化选项

2021.4.27

problem fixed

2021.10.6

重启研究，目前无线通信环境配置部分已经完成

开始强化学习部分的配置

2021.10.11

buildState部分的代码仍然有问题，需要重新把复数展开成Amplitude & Phase或者Real & Image

Reward Function没有加Interference Penalty

代码主题部分写完了，但是有大bug。网络训练之后，出来的决策和随机的吗没有任何区别！！！

网络输出的Q值太小了

现在发现了channel有bug，明天调试

明天计划重新复习强化学习