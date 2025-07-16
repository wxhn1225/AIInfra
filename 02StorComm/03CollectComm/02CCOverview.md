<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 为什么需要集合通信

Author by: SingularityKChen

本章内容聚焦集合通信的基础知识和相关术语。

## AI 集合通信背景

本节简要回顾神经网络及其训练过程的基础计算，并回顾近年来 AI 训练和推理过程中涉及的分布式和并行计算模式。

### 单卡到多卡

### 多卡训练与并行

介绍多卡训练的几种并行方式及其对应的 CCL

#### 流水并行 Pipeline Parallelism

- Send/Recv

#### 数据并行 Data Parallelism

- AllReduce

#### 张量并行 Tensor Parallelism

- AllGather

#### 多维并行 Multi Parallelism

- MoE(Mixture of Experts)
  - Send/Recv(all2all)
- FSDP(Full Sharded DP)
  - AllGather
- Long Sequence
  - AllGather/AllReduce

<!-- ## 为什么需要集合通信算法

- 通信算法
- 通信原语/操作
- 拓扑算法很多，但不是所有拓扑算法都能满足实际生产需求，需要具体问题具体分析、具体场景具体设计，因此出现了XCCL -->

## XCCL 基本架构

本节从 HPC 和 XCCL 通信架构对比介绍，展示二者的异同。

### 计算与通信解耦

神经网络训练过程中，每一层神经网络都会计算出一个梯度Grad，如果反向传播得到一个梯度马上调用集合通信AllReduce 进行梯度规约，在集群中将计算与通信同步串行，那么集群的模型算力利用率（MFU）就很低。

将计算与通信解耦，计算的归计算，通信的归通信，通过性能优化策略减少通信的次数：
- 提升集群训练性能（模型利用率 MFU/算力利用率MFU）；
- 避免通信与计算假死锁（计算耗时长，通信长期等待）；

### 分布式加速库

分布式加速库：解耦计算和通信，分别提供计算、通信、内存、并行策略的优化方案。
- DeepSpeed、Megatron-LM、MindSpeed、ColossalAI

### HPC 到 AI 通信栈基本架构

### XCCL 在 AI 系统中的位置

- Megatron-LM/MindSpeed 分布式加速框架解耦了计算与通信：
1. 计算主要通过PyTorch 等AI框架执行；
2. 通信通过XCCL 通信库来执行；

将框架计算出来 Tensor 记录到 Bucket 中
- 通过控制层在后台启动loop 线程
- 周期性的从Bucket 中读取Tensor
- 控制层在节点之间协商一致后，进行消息分发到具体 NPU 上执行通信

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?aid=1255396066&bvid=BV18J4m1G7UU&cid=1570235726&page=1&as_wide=1&high_quality=1&danmaku=0&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>
</html>