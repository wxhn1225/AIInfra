***Megatron-Core：PP 基本原理（朴素流水并行原理，Bubble空泡率计算，Gpipe原理解析与动态内存峰值分析）***

**1）朴素流水并行原理：**
    朴素流水并行原理较为简单，如图所示：四种不同的色块代表不同的rank（或GPU），rank0将本地计算后的激活值传递给后续的rank1，以此类推直到rank3完成类似流水线的前向传输过程，此时rank3开启反向流水线计算过程，将本阶段的计算梯度传递给rank2，以此类推完成反向传输过程，最终得到各个rank的梯度，进行模型各个流水线阶段的权重参数更新，此时完成一轮流水线迭代训练。
![alt text](image.png)
    
  **2）Bubble空泡率计算：**
    此时显然可见，图中大部分的时间为空白，计算与通信缺乏overlap，即存在计算等通信的现象（rank1要等待rank0激活前向传递之后才能计算），因此对于图中的“空白部分”，我们引入Bubble的概念来定量的评估流水线并行的性能。
    空泡 Bubble 的产生，是因为算子内并行和算子间并行解决单设备内存不足的问题，模型经过拆分后的上一个 rank 的 stage 需要长期持续处于空闲状态，等待其他 rank 的 stage 计算完成，才可以开始计算，这极大降低了设备的平均使用率。这种现象被称为并行空泡（Parallelism Bubble）。
    总的 bubble 占用的时间跟流水并行 PP 切分策略相关：
      $$\begin{equation}
t_{bubble} = (p - 1)(t_f + t_b)
\end{equation}$$
      其中 $p$为并行度，$t_f$为前向时间，$t_b$ 为反向时间，pipeline bubble占据了 $(p-1)$个前向、反向过程。
    Bubble 占有率比例bubbletaion，又称空泡率，计算由公式给出：
    $$\begin{equation}
\mathit{bubble ration} = \frac{t_{bubble}}{t_{bubble} + t_{ideal}} 
= \frac{(p-1)(t_f + t_b)}{(p-1)(t_f + t_b) + m(t_f + t_b)} 
= \frac{p - 1}{m + p - 1}
\end{equation}$$
    其中 $t_{ideal}$为理想迭代时间，$m$为 micro-batch的数量，$t_f$，$t_b$ 为单个$micro-batch$ 时间，因此$t_{ideal}=m(t_f+t_b)$。
    根据上面的公式，$bubble ration$跟 $micro-batches$ 有关系，micro-batch数量（m）越多，Bubble 的比例越会降低到可接受的水平，因此在衡量大模型性能优化的过程，$bubble ration$ 是作为一个衡量指标去看待其利用率。

  **3）Gpipe原理解析：**
    Gpipe是基于上述特点推出的流水线并行技术：将一个batch size的数据切分成四个micro-batch size，每个micro-batch作为朴素流水并行方式中的一个batch，前向过程从rank0流向rank3（又称为warmup），再反向回溯（称为cooldown）。
    ![alt text](Gpipe.png)
    计算空泡率：
    $$\begin{equation}
bubble ration=\frac{t_{bubble}}{t_{ideal}}=\frac{p-1}{m}
\end{equation}$$
    因此为降低空泡率，通常需要增加数据切分micro-batches的数量m，即令$m>>p$.
    在模型的反向传输过程中，由于GPU需要保存前向传播时的中间激活值，以便计算梯度，因此划分micro-batch的数目m将由单GPU计算卡的显存约束（eg. 相同色块为一张GPU,对于GPU1来说需要保存m=4个前向过程的激活值，因此当使用多个 micro-batch，激活值存储量线性增加），在warmup阶段结束后所有GPU显存，达到称为动态内存峰值。

**4）动态内存峰值分析：**
    为解决Gpipe带来的动态内存峰值问题，重计算（Recomputation）技术被引入解决显存瓶颈问题，其核心思想为：与其在前向传播中缓存所有中间激活值，不如在反向传输时“重新计算”一遍前向过程，来获得需要的激活值，从而节省显存。
    (待补充)