> 2022 Spring HUSTAIA

# 基于EEG的运动想象状态分类

## 数据采集过程：

采集过程中，受试者坐在电脑前的椅子上。采集开始时，电脑屏幕上会出现一个固定的叉，提示对象准备，持续3s；然后，一个指向某一个方向的箭头作为视觉提示在屏幕上出现5s，在此期间，受试者根据箭头的方向执行特定的运动想象任务；然后，视觉提示从屏幕上消失，受试者短暂休息2s。紧接着下一个trial开始。

## 数据集：

数据来自8个健康的受试者（训练受试者S1～S4，测试受试者S5～S8），每一个受试者执行两类运动想象任务：右手和双脚，脑电信号由一个13通道的脑电帽以512Hz的频率记录得到。我们提供了经过预处理后的数据：下采样到了250Hz，带通滤波至8-32Hz，划分每一次视觉提示出现后的0.5~3.5s之间的EEG信号作为一个trial。每个用户包含200个trial（右手和双脚各100个trial）。

数据以.npz和.mat格式提供，包含：

- X: 预处理后的EEG信号,  维度: [trails * 通道* 采样点]。
- y: 类别标签向量。测试数据不包含此变量。


## 实验算法
- preprocess
    - euclid align
    - riemann align
- FBCSP
    - filter bank+CSP+MIBIF feature select
- CSP
    - maen convariance calculated using euclid, logeuclid, riemann distance
- Tangent Space
    - TangentSpace mapping+anova feature select
- BrainNetv1
    - based on EEGNet(conv+pool)
- BrainNetv2
    - conv+pool+self attention
- BrainNetv3
    - conv+pool+snn

## 实验结果
> best result

cross validation method: Three person as training set, one person as validate set

| preprocess | TS+LDA        | CSP+SVM       | FBCSP+SVM              | BrainNetv1    | BrainNetv2    | BrainNetv3    |
|------------|---------------|---------------|------------------------|---------------|---------------|---------------|
| orig       | 0.59+/-0.07 | 0.63+/-0.12 | **0.65+/-0.10**       | 0.61+/-0.09 | 0.49+/-0.02 | 0.52+/-0.02 |
| ea         | 0.57+/-0.12 | 0.62+/-0.09 | **0.63+/-0.08** | 0.62+/-0.10 | 0.52+/-0.02 | 0.55+/-0.02 |
| ra         | 0.58+/-0.07 | 0.61+/-0.08 | **0.64+/-0.08** | 0.61+/-0.13 | 0.53+/-0.02 | 0.54+/-0.05 |