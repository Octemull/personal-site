---
title: "chap 09 - 聚类 | Clustering"
date: 2017-12-14
lastmod: 2019-04-01
draft: false
show_comments: true
keywords: []
description: ""
tags: [Notes]
categories: [Machine Learning]
---

![D1EB9267-94FB-43A9-8113-A77F95ADA1DE](https://i.loli.net/2019/04/01/5ca220a094654.png)


## 9.1 聚类任务 clustering

**类别：**无监督学习 (unsupervised learning)

**常见的无监督学习任务：** 

聚类(clustering)、密度估计(density estimation)、异常检测(anomaly detection)

**聚类：**将样本集划分为若干个通常是不相交的子集，每个子集称为一个“簇”(cluster)。每个簇可能对应某个潜在（事先不知道，需要聚类后命名）的类别。如对西瓜聚类，可能得到“浅色瓜”“深色瓜”“外地瓜”“本地瓜”等。

**聚类的数学表示：**

假定样本集$D = \{\boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_m\}$包含$m$个无标记样本，每个样本$\boldsymbol{x} = \{x_1;x_2; \cdots,x_n \}$是一个$n$维特征向量，则聚类算法将样本集$D$划分为$k$个不相交的簇$\{C_l \, | \, l=1,2,\cdots, k\}$其中$C_{l'} \cap_{l' \neq l}C_l = \varnothing$且$D = \cup_{l=1}^k C_l$. 相应地，用$\lambda_j \in \{1,2,\cdots , k\}$表示样本$\boldsymbol{x}_j$的“簇标记”(cluster label)，即$\boldsymbol{x}_j \in C_{\lambda_j}$.于是，聚类的结果可用包含$m$个元素的簇标记向量$\boldsymbol{\lambda}=(\lambda_1; \lambda_2; \cdots, \lambda_m)$表示。

**聚类的适用场景：**

* 可作为单独过程，寻找数据内在的分布结构；
* 也可作为分类任务的先驱过程。

**举一个例子：**

商业应用中先对顾客进行聚类后，把顾客分为几个类型。然后用分类后的数据做训练集训练分类器，等有新顾客来的时候就能判断新顾客的类型。

## 9.2 性能度量

**聚类性能度量：**有效性指标(validity index)

**好的聚类：**“物以类聚”。聚类结果的“簇内相似度”(intra-cluster similarity)高且“簇间相似度”(inter-cluster similarity)低。

**分类：**

* **外部指标 external index**：将聚类结果与某个参考模型(reference model)比较【如，将领域专家划分结果作为参考模型】；
* **内部指标 internal index**：直接考察聚类结果。

### 外部指标
 
对数据集$D = \{\boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_m\}$，假定通过聚类给出的划分结果为$\cal{C}=\{C_1, C_2, \cdots, C_k\}$, 参考模型给出的簇划分为$\cal{C}^\ast = \{C_1^\ast, C_2^\ast, \cdots, C_s^\ast\}$. 相应，令$λ$与$λ^\ast$分别表示与$C$和$C^\ast$对应的簇标记向量。将样本两两配对考虑，定义
![D7BADEB8-39E2-4DBF-8E8A-D436CCC3D6B](https://i.loli.net/2019/04/01/5ca220c18a902.png)

**说明：**

* a 集合SS：包含在C中隶属于相同簇，且在C*中也隶属于相同簇的样本对；
* b 集合SD：包含在C中隶属于相同簇，但在C*中隶属于不同簇的样本对；
* c 集合DS：包含在C中隶属于不同簇，但在C*中隶属于相同簇的样本对；
* d 集合DD：包含在C中隶属于不同簇，且在C*中也隶属于不同簇的样本对。

由上述式子，可导出常用外部指标：

* Jaccard系数(Jaccard Coeffcient, JC)
![043E46FD-5619-474B-9661-96B544FFCFFA](https://i.loli.net/2019/04/01/5ca2209469d2d.png)

* FM指数(Foelkes and Mallows Index, FMI)![3E3D1748-7D0E-4E70-9854-4CE74D9CA363](https://i.loli.net/2019/04/01/5ca220946b76e.png)


* Rand指数(Rand Index, RI)
![10701FCA-F894-4181-A746-A53811FD7961](https://i.loli.net/2019/04/01/5ca220946d20c.png)

**指标范围：**[0, 1]

**判断方法：**值越大，聚类性能越好。


### 内部指标

考虑聚类结果的簇划分$\cal{C}=\{C_1, C_2, \cdots, C_k\}$，定义
![07F3807C-9270-4D02-80CF-5278601127A0](https://i.loli.net/2019/04/01/5ca220c148507.png)

**说明：**

* $dist(·,·)$：两个样本之间的距离；
* $μ$：簇C的中心点$\boldsymbol{\mu} = \frac{1}{|C|} \sum_{i \leq i \leq |C|} \boldsymbol{x}_i$；
* $avg(C)$：簇$C$内样本间的平均距离（共有$|C|(|C|-1)/2$个距离值）；
* $diam(C)$：簇$C$内样本间的最远距离；
* $d_{\min}(Ci,Cj)$：簇$C_i$与簇$C_j$最近样本间距离；
* $d_{cen}(Ci,Cj)$：簇$C_i$与簇$C_j$中心点之间的距离。

由此，可得常用内部指标：

* DB指数(Davies-Bouldin Index, DBI)
![DCD1E211-9C89-4AB6-9E18-43A242339B88](https://i.loli.net/2019/04/01/5ca220b7c2dbc.png)

* Dunn指数(Dunn Index, DI)
![EF100FA8-3C74-4EF2-8A78-1DD1A7617C](https://i.loli.net/2019/04/01/5ca220b1a4b32.png)

**判断方法：**

* DBI越小，聚类性能越好；
* DI越大，聚类性能越好。

## 9.3 距离计算 dist(·,·)

**距离度量的基本性质：**

* 非负性：$dist(\boldsymbol{x}_i, \boldsymbol{x}_j) \geq 0 \tag{9.14}$
* 同一性：$dist(\boldsymbol{x}_i, \boldsymbol{x}_j) = 0 \text{ 当且仅当 } \boldsymbol{x}_i = \boldsymbol{x}_j\tag{9.15}$
* 对称性：$dist(\boldsymbol{x}_i, \boldsymbol{x}_j) =dist(\boldsymbol{x}_j, \boldsymbol{x}_i)  \tag{9.16}$
* 直递性(三角不等式)：$dist(\boldsymbol{x}_i, \boldsymbol{x}_j) \leq dist(\boldsymbol{x}_i, \boldsymbol{x}_k) + dist(\boldsymbol{x}_k, \boldsymbol{x}_j)   \tag{9.17}$

**属性分类：**

通常将属性划分为“连续属性”(continuous attribute)【在定义域上有无穷多个可能取值】和“离散属性”(categorical attribute)【在定义域上只有有限个取值】。

> - 连续属性亦称“数值属性”（numerical attirbute），离散属性亦称“列名属性”(nominal attribute)。
> - 定义距离时，考虑有序和无序更重要。

* 有序属性：属性值之间有次序，可直接在属性值上计算距离。
例如，定义域为{1,2,3}的离散属性与连续属性更相似，1与2较近、1与3较远。
* 无序属性：属性值之间无次序，不能直接用属性值计算距离。
例如，定义域为{飞机，火车，轮船}。

### 闵可夫斯基距离

**适用：**有序属性

给定样本$\boldsymbol{x}_i = \{x_{i1};x_{i2}; \cdots ; x_{in} \}$与$\boldsymbol{x}_j = \{x_{j1};x_{j2}; \cdots ; x_{jn} \}$，闵式距离定义如下
![86F3F30C-7DB9-430D-8B97-44BB2F9390D1](https://i.loli.net/2019/04/01/5ca2209496aab.png)

$p≥1$时，显然满足距离度量的基本性质。【上式即为$\boldsymbol{x}_i - \boldsymbol{x}_j$的$L_p$范数$||\boldsymbol{x}_i - \boldsymbol{x}_j||_p$】

* p=2，“欧氏距离”(Euclidean Distance)
![F673D9FA-CFC1-4223-ABD8-23189F2CE67](https://i.loli.net/2019/04/01/5ca220b1a31c9.png)

* p=1，“曼哈顿距离”(Manhattan distance)
![C6523444-6BF2-4A33-9245-F2191CF73A30](https://i.loli.net/2019/04/01/5ca220b7c1397.png)

> 亦称“街区距离”（city block distance）

* p→∞，“切比雪夫距离”（）


### VDM（Value Difference Metric）[Stanfill and Waltz, 1986]

**适用：**无序属性

**符号：**

* $m_{u,a}$：在属性$u$上取值为$a$的样本数；
* $m_{u,a,i}$：在第$i$个样本簇中在属性$u$上取值为$a$的样本数；
* $k$：样本簇数。【样本类别已知时k通常设置为类别数。】

属性$u$上两个离散值$a$与$b$之间的VDM距离为
![E7105DF5-26CC-4E82-B470-35ED3ED0D96](https://i.loli.net/2019/04/01/5ca220b1a1703.png)



### 闵式距离&VDM

**适用：**混合属性

假设有$nc$个有序属性、$n-n_c$个无序属性，不失一般性，令有序属性排列在无序属性之前，则
![1EF9E398-42BC-48BD-81E7-6321E7DCE4DB](https://i.loli.net/2019/04/01/5ca220b7dfe99.png)

### 加权距离(weighted distance)

**适用：**属性重要程度不同时

**例子：**加权闵式距离
![424EC9B5-D2FE-4786-B077-728031AC096](https://i.loli.net/2019/04/01/5ca220b182f2d.png)

其中，权重$w_i \geq 0 \, (i=1,2,\cdots, n)$表征不同属性的重要性，通常$\sum_{i=1}^n w_i =1$.

### 非度量距离 non-metric distance

**适用：**不满足直递性的相似度度量距离

**例子：**“人”“马”分别与“人马”相似，但“人”与“马”很不相似；在距离上表现为，“人”“马”分别与“人马”的距离都较小，“人”与“马”的距离很大。如图9.1所示。
![8AB64789-0A67-4744-9762-0C4603D57E43](https://i.loli.net/2019/04/01/5ca220c730d8d.png)

> 该例子中，从数学上看，令$d_3=3$即可满足直递性；但从语意上看，$d_3$应远大于$d_1$与$d_2$。

## 9.4 原型聚类

原型聚类亦称“基于原型的聚类”(prototype-based clustering)

* **假设**：聚类结构能通过一组原型刻画（“原型”指样本空间中具有代表性的点）
* **基本思路：**先对原型进行初始化，然后对原型进行迭代更新求解。
* **区别：** 初始化方法不同，求解方式不同。

### 9.4.1 k均值算法 (k-means)

**符号：**

* 样本集：$D = \{\boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_m\}$
* 聚类所得簇划分：$\cal{C}=\{C_1, C_2, \cdots, C_k\}$
* 簇𝐶𝑖的均值向量：$\boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{\boldsymbol{x} \in C_i} \boldsymbol{x}$

**目标：**k-means算法对聚类所得簇划分最小化平方误差
![1C996A78-046C-4668-A819-C6F3FC68CB78](https://i.loli.net/2019/04/01/5ca2209498b7d.png)

**说明：**

* E在一定程度上刻画了簇内样本围绕簇均值向量的紧密程度；
* E越小则簇内样本相似度越高。

**难点：**最小化平方误差E是NP难问题，需考察对样本集的所有划分。

**解决：**贪心策略，迭代优化得近似解。

**算法流程：**
![2811345E-890D-497A-B99B-A0C4EF232DFB](https://i.loli.net/2019/04/01/5ca220a1b13c8.png)

* 1：对均值向量进行初始化；
* 4-8、9-10：对当前簇划分及均值向量迭代更新，若迭代更新后聚类结果不变，则在第18行返回当前划分结果。

> 为避免运行时间过长，可设置最大迭代次数限制或最小调整幅度阈值，一旦超过最大迭代次数或调整幅度小于阈值，则停止运行。

**举一个例子：**

以表9.1的西瓜数据集4.0为例。将编号为$i$的样本称作$\boldsymbol{x}_i$，$\boldsymbol{x}_i$是一个二维向量。
![3BA1CC0E-1C5C-472C-8B20-6D8711400A78](https://i.loli.net/2019/04/01/5ca220a0f1f60.png)

假设聚类簇数$k=3$，开始随机选取三个样本$\boldsymbol{x}_6$，$\boldsymbol{x}_{12}$，$\boldsymbol{x}_{24}$作为初始均值向量，即
![299DDB6B-86A8-4B7C-A78E-2B06A1911C03](https://i.loli.net/2019/04/01/5ca220b7e2292.png)

考察样本$\boldsymbol{x}_1=(0.697; 0.460)$，它与当前均值向量𝛍1,𝛍2,𝛍3的距离分别为0.369, 0.506, 0.220. 因此把$\boldsymbol{x}_1$划分到簇C3中。类似，对数据集中所有样本考察依次按后，可得当前簇划分为
![79698B59-5181-4653-AD1A-442EAFF205AE](https://i.loli.net/2019/04/01/5ca220b184a62.png)
于是，可从C1、C2、C3分别求出新的均值向量
![8305A13F-7459-4DE8-A369-7F4F28DC3E2E](https://i.loli.net/2019/04/01/5ca2209493523.png)
更新当前均值向量后，不断重复上述过程，如图9.3所示，第五轮迭代产生的结果与第四轮迭代相同，于是算法停止，得到最终的簇划分。
![59A94059-798B-4E14-BAB4-068FDCD68659](https://i.loli.net/2019/04/01/5ca220a16d147.png)

### 9.4.2 学习向量量化 (Learning Vector Quantization, LVQ)

**区别：**假设数据样本带有类标记，学习过程利用样本的标记信息（监督信息）来辅助聚类。

**符号：**

* $D = \{\boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_m\}$：样本集；
* $(x_{j1};x_{j2}; \cdots; x_{jn})$：每个样本$\boldsymbol{x}_j$是由$n$个属性描述的特征向量;
* $𝑦_𝑗 ∈ 𝜸$：样本$\boldsymbol{x}_j$的类别标记；
* $η$：学习率。

**目标：** 习得一组$n$维原型向量$\{\boldsymbol{p}_1, \boldsymbol{p}_2, \cdots, \boldsymbol{p}_q\}$每个原型向量代表一个聚类簇，簇标记为$t_i \in \cal{Y}$.

**算法：**
![28ACBF90-DEC7-4BF2-946E-1225EB398ADE](https://i.loli.net/2019/04/01/5ca220a100482.png)

* 1：初始化原型向量（如，对第q个簇可从类别标记为$t_q$的样本中随机选取一个作为原型向量）；
* 2-12：对原型向量迭代优化；
* 每轮迭代中，随机选取一个有标记的训练样本，找到与其距离最近的原型向量，并根据连着挂这两者的类别标记是否一致来对原型向量进行相应更新。
* 5：竞争学习的“胜者为王”策略。SOM是基于无标记样本的聚类算法，而LVQ可看做SOM基于监督信息的扩展。竞争学习与SOM参见5.5.2和5.5.3.
* 7：$\boldsymbol{x}_j$与$𝒑_{𝑖^\ast}$的类别相同；
* 9：$\boldsymbol{x}_j$与$𝒑_{𝑖^\ast}$的类别不同；
* 12：若算法停止条件已满足，则将当前原型向量作为最终结果返回。

**如何更新原型向量（算法第6-10行）**

若当最近的原型向量$𝒑_{𝑖^\ast}$与$\boldsymbol{x}_j$的类别标记相同，则令$𝒑_{𝑖^\ast}$向$\boldsymbol{x}_j$的方向靠拢（第7行）
![B7DD6923-C389-4927-B38E-4E5FD05C9DDB](https://i.loli.net/2019/04/01/5ca220949f53a.png)

𝒑'与$\boldsymbol{x}_j$间的距离为
![553FA199-179F-41D5-A8C3-D8B8913F4861](https://i.loli.net/2019/04/01/5ca220b7c47c4.png)

反之，若类别标记不同，则令$𝒑_{𝑖^\ast}$远离$\boldsymbol{x}_j$的方向（第9行）
𝒑'与$\boldsymbol{x}_j$间的距离增大为$(1 + \eta)\cdot ||𝒑_{𝑖^\ast} - \boldsymbol{x}_j||_2$。

**Voronoi剖分 (Voronoi tessellation)**

如此，习得一组原型向量$\{\boldsymbol{p}_1, \boldsymbol{p}_2, \cdots, \boldsymbol{p}_q\}$后即可对样本空间𝓧进行簇划分。将样本𝒙划分至与其最近的原型向量代表的簇中。每个原型向量$𝒑_𝑖$定义了与之相关的区域$𝑅_𝑖$，该区域中每个样本与$𝒑_𝑖$的距离不大于它与其他原型向量$𝒑_{𝑖'}(𝑖'≠𝑖)$的距离，即
![C96733C6-C089-4F86-B9FE-4EDC712148AA](https://i.loli.net/2019/04/01/5ca22094a4699.png)
由此形成的对样本空间𝓧的簇划分${𝑅_1,𝑅_2,…,𝑅_q}$，该划分通常称为“Voroni剖分”。

> 若将中样本全用用原型向量表示，则可实现数据的“有损压缩”(lossy compression)，这称为“向量量化”(vector quantization)；LVQ由此得名。

**举一个例子🌰：**

以表9.1的西瓜数据集4.0为例。

令9-21号样本的类别标记为c2，其他样本的类别标记为c1。假定q=5，即学习目标是找到5个原型向量$\boldsymbol{p}_1, \boldsymbol{p}_2, \cdots, \boldsymbol{p}_5$并假定其对应的类别标记分别为$c_1, c_2, c_2, c_1, c_1$(即，希望为“好瓜=是”找到3个簇，“好瓜=否”找到2个簇。)

算法开始时，根据样本的类别标记和簇的预设类别标记对原型向量进行随机初始化，假定初始化为样本$\boldsymbol{x}_5, \boldsymbol{x}_{12}, \boldsymbol{x}_{18}, \boldsymbol{x}_{23}, \boldsymbol{x}_{29}$.在第一轮迭代中，假定随机选取的样本为$𝒙_1$，该样本与当前原型向量的距离分别为0.283, 0.506, 0.434, 0.260, 0.032. 由于$𝒑_5$与$𝒙_1$距离最近且类别标记相同(c2)，假定学习率$η=0.1$，则LVQ更新$𝒑_5$得到新原型向量
![4D3591EE-B0BF-4061-963C-4A800EA721AB](https://i.loli.net/2019/04/01/5ca220c04423d.png)
将$𝒑_5$更新为$𝒑'$后，不断重复上述步骤，不同轮数之后的聚类结果如图9.5所示。
![56E4C966-2DD6-47D7-A21B-F218EDE24A72](https://i.loli.net/2019/04/01/5ca220a4178df.png)


### 9.4.3 高斯混合聚类 (Mixture-of-Gaussian)

**区别：**采用概率模型来表达聚类原型。

**（多元）高斯分布**

对n维样本空间𝓧找那个的随机向量$\boldsymbol{x}$，若$\boldsymbol{x}$服从高斯分布，其概率密度函数为
![8DFB54DA-5DCF-4C41-9525-F7EE00B9](https://i.loli.net/2019/04/01/5ca220b1a654d.png)

记做$𝒙\sim \cal{N}(𝛍,𝚺)$. $𝚺$：对称正定矩阵；$|𝚺|$：$𝚺$的行列式；$𝚺^{-1}$：$𝚺$的逆矩阵。

**符号：**

* 𝛍：n维均值向量；
* 𝚺：n×n协方差矩阵。

高斯分布完全由上述两个参数决定，把其概率密度函数记做$p(\boldsymbol{x} \, | \, 𝛍,𝚺)$

#### 高斯混合分布
![3E8A58F4-26D5-4579-ADEB-9705F7FDBF5B](https://i.loli.net/2019/04/01/5ca220b7d6d56.png)

> $𝑝_{\cal{M}}(·)$也是概率密度函数，$\int p_{\cal{M}}(\boldsymbol{x}) d\boldsymbol{x} =1$.

共由$k$个混合成分组成，每个混合成分对应一个高斯分布。其中$𝛍_𝑖$与$𝚺_𝑖$是第$i$个高斯混合成分的参数，而$𝛼_𝑖>0$为相应的“混合系数”(mixture coefficient)，$\sum_{i=1}^k \alpha_i = 1$.

**算法：**
![FCE21442-A5F3-41F8-A9C8-6CA3B189](https://i.loli.net/2019/04/01/5ca220a1f316f.png)

**思路：**

假设样本的生成过程由高斯混合分布给出：

* 首先，根据$\alpha_1,\alpha_2,\cdots,\alpha_k$定义的先验分布选择高斯混合成分，其中$𝛼_𝑖$为选择第$i$个混合成分的概率；
* 然后，根据被选择的混合成分的概率密度函数进行采样，生成相应的样本。

#### 后验概率和先验概率

若训练集$D = \{\boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_m\}$由上述过程生成，令随机变量$z_j \in \{1,2,\cdots,k\}$表示生成样本$\boldsymbol{x}_j$的高斯混合成分，其取值未知。

$z_j$的**先验概率**$P(z_j=i)$对应于$α_i(i=1,2,...,k)$。根据贝叶斯定理，$z_j$的后验分布对应于
![03486FFE-73B1-470D-88C0-9D1E563BE07E](https://i.loli.net/2019/04/01/5ca220c0424aa.png)


即，$p_{\cal{M}}(z_j=i \, | \, \boldsymbol{x}_j)$给出了样本$\boldsymbol{x}_j$由第$i$个高斯混合成分生成的后验概率，记做$γ_{ji}\, (i=1, 2, ..., k)$。

**确定样本所属簇标记λj**

当高斯混合分布(9.29)已知时，高斯混合聚类将把样本集$D$划分为$k$个簇$\cal{C}=\{C_1, C_2, \cdots, C_k\}$每个样本的簇标记$λ_j$如下确定
![71577F1C-EC81-43E5-9B57-6CEDDD64CBEA](https://i.loli.net/2019/04/01/5ca22094ac43d.png)

{{% admonition tip summary %}}
从原型聚类的角度看，高斯混合聚类是采用概率模型（高斯分布）对原型进行刻画，簇划分则由原型对应的后验概率决定。
{{% /admonition %}}

#### 参数求解
 $$\{ (\alpha_i, \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i \, | \, 1 \leq i \leq k\}$$

**方法：**极大似然法+EM算法

给定样本集$D$，用极大似然估计，最大化（对数）似然
![C75EA0EB-A3AB-489B-88E2-0DE3638C9D](https://i.loli.net/2019/04/01/5ca220c04f053.png)

用EM算法迭代优化求解：

**求样本均值向量μ_i**

若参数$\{ (\alpha_i, \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i \, | \, 1 \leq i \leq k\}$能使式(9.32)最大化，则由$\frac{\partial LL(D)}{\partial \boldsymbol{\mu}_i} =0$有
![84C4A559-65EC-470E-AF81-BF4885A78552](https://i.loli.net/2019/04/01/5ca220b7d522a.png)

由式(9.30)以及$\gamma_{ji} = p_{\cal{M}} (z_j = i \, | \, \boldsymbol{x})$有
![F53B4D27-5BC3-411C-AD0F-44AEFC5EE7A2](https://i.loli.net/2019/04/01/5ca220c051c89.png)

即**各混合成分的均值可通过样本加权平均来估计，样本权重是每个样本属于该成分的后验概率。**

**求协方差阵$∑_i$**

类似，令$\frac{\partial LL(D)}{\partial \boldsymbol{\Sigma}_i} =0$可得
![05F16BEE-6735-4086-8C20-634C76C327F3](https://i.loli.net/2019/04/01/5ca220c0adbe7.png)

**求高斯成分的混合系数$α_i$**

对混合系数$α_i$，除了最大化$LL(D)$，还需满足考虑$LL(D)的$拉格朗日形式
![F9F06440-743A-4A24-9684-FB64D682974B](https://i.loli.net/2019/04/01/5ca220b18e276.png)

其中$λ$为拉格朗日乘子。由式(9.36)对$α_i$的导数为$0$，有
![1BD63CB7-C78B-4021-8BA1-DBEC05E96940](https://i.loli.net/2019/04/01/5ca220b7cda1c.png)

两边同乘以$α_i$，对所有混合成分求和可知$λ=-m$，有
![8C5ADF68-170D-409B-9B97-C98F4E1EEC8A](https://i.loli.net/2019/04/01/5ca220b7cb23e.png)

即，**每个高斯成分的混合系数由样本属于该成分的平均后验概率决定。**


#### 高斯混合模型的EM算法：

* E：在每步迭代中，先根据当前参数来计算每个样本属于每个高斯成分的后验概率$γ_{ji}$；
* M：根据均值向量、协方差阵、混合系数的公式来更新模型参数$\{ (\alpha_i, \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i \, | \, 1 \leq i \leq k\}$。

**举一个例子🌰：**

以表9.1的西瓜数据集4.0为例，令高斯混合成分的个数$k=3$。算法开始时，假定初始化模型参数为
![-w588](https://i.loli.net/2019/04/01/5ca220b192a22.jpg)
第一轮迭代，先计算样本由3个混合成分生成的后验概率。以x1为例，由式(9.30)算出后验概率
![-w592](https://i.loli.net/2019/04/01/5ca220949738f.jpg)
所有样本的后验概率算法后，得到新的模型参数如下：
![-w590](https://i.loli.net/2019/04/01/5ca220c0288bf.jpg)
更新参数，不断重复上述过程，不同轮数之后的聚类结果如图9.7所示。
![C296E6DC-A0CB-457A-93B0-E8AC9CD480E8](https://i.loli.net/2019/04/01/5ca220c7a7784.png)

## 9.5 密度聚类

> 亦称“基于密度的聚类”(density--based clustering)

**思路：**假设聚类结构能通过样本分布的紧密程度决定。通常情形下，魔都聚类算法从样本密度的角度来考察样本之间的可连接性，并基于可连接样本不断扩展聚类簇以获得最终结果。

**代表算法：** DBSCAN (Density-Based Spatial Clustering of Applications with Noise), 基于一组“领域”(neighborhood)参数$(𝜖, MinPts)$来刻画样本分布的紧密程度。

**符号：**

给定数据集$D = \{\boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_m\}$，定义

* **$𝜖-$邻域**：对$\boldsymbol{x}_j$∈D，其𝜖-领域包含样本集D中与$\boldsymbol{x}_j$的距离不大于𝜖的样本，即$N_{\epsilon}(\boldsymbol{x}_j) = \{\boldsymbol{x}_i \in D \, | \, dist(\boldsymbol{x}_i, \boldsymbol{x}_j) \leq \epsilon$
* **核心对象(core object)**：若$\boldsymbol{x}_j$的𝜖-邻域至少包含$MinPts$个样本，即$N_{\epsilon}(\boldsymbol{x}_j) \geq MinPts$，则$\boldsymbol{x}_j$是一个核心对象；
* **密度直达(deirectly density-reachable)**：若$\boldsymbol{x}_j$位于$\boldsymbol{x}_1$的𝜖-邻域中，且$\boldsymbol{x}_i$是核心对象，则称$\boldsymbol{x}_j$由$\boldsymbol{x}_i$密度直达；（通常不满足对称性）
* **密度可达(density-reachable)**：对𝒙𝑖与$\boldsymbol{x}_j$，若存在样本序列$𝒑_1,𝒑_2,...,𝒑_n$, 其中$𝒑_1=𝒙_𝑖$, $𝒑_n=\boldsymbol{x}_j$且$𝒑_{𝑖+1}$由$𝒑_𝑖$密度直达，则称$\boldsymbol{x}_j$由$\boldsymbol{x}_i$密度可达；（满足直递性，但不满足对称性）
* **密度相连(density-connected)**：对$\boldsymbol{x}_i$与$\boldsymbol{x}_j$，若存在$\boldsymbol{x}_k$使得$\boldsymbol{x}_i$与$\boldsymbol{x}_j$均由$\boldsymbol{x}_k$密度可达，则称$\boldsymbol{x}_i$与$\boldsymbol{x}_j$密度相连。（满足对称性）

直观图如下：
![D8CAAD56-FBE1-47BD-AED9-60AA4764F6A9](https://i.loli.net/2019/04/01/5ca220c72eb1d.png)

#### 簇

**定义：**由密度可达关系导出的最大的密度相连样本集合。用数学语言表达，给定领域参数$(𝜖, MinPts)$，簇$𝐶⊆𝐷$是满足一下性质的非空样本子集：

* 连接性：$𝒙_𝑖∈𝐶$，$\boldsymbol{x}_j ∈𝐶$ ⇒  $𝒙_𝑖$与$\boldsymbol{x}_j$密度相连                                              (9.39)
* 最大性：$𝒙_𝑖∈𝐶$，$\boldsymbol{x}_j$由$𝒙_𝑖$密度可达  ⇒  $\boldsymbol{x}_j∈𝐶$                                             (9.40)

> D中不属于任何簇的样本被认为是噪声noise或异常anomaly样本。

**寻找聚类簇**

**聚类簇： * 若$𝒙$为核心对象，由$𝒙$密度可达的所有样本组成的集合记为$X=\{𝒙'∈D \, | \, 𝒙'由𝒙密度可达\}$，则X即为满足连接性与最大性的簇。

**思路：**先任选数据集中的一个核心对象为“种子”(seed)，再由此出发确定相应的聚类簇。

**算法描述：**
![1DC9D45F-5085-4465-8359-220DE4225618](https://i.loli.net/2019/04/01/5ca220a200e81.png)

**说明：**

* 1-7：根据给定的邻域参数(𝜖, MinPts)找出所有核心对象；
* 10-24：以任一对象为出发点，找出由其密度可达的样本生成聚类簇，直到所有核心对象均被访问过为止。

**举一个例子🌰：**

以表9.1的西瓜数据集4.0为例。 假定邻域参数$(𝜖, MinPts)$设置为$𝜖=0.11$，$MinPts=5$.

1. 先找出各样本的𝜖-邻域并确定核心对象集合：
![-w604](https://i.loli.net/2019/04/01/5ca220b1967b0.jpg)
2. 然后，从Ω中随机选取一个核心对象作为种子，找出由它密度可达的所有样本，这就构成了第一个聚类簇。不妨假定核心对象$𝒙_8$被选做种子，则DBSCAN生成的第一个聚类簇为
![1863525D-5D66-4B6C-A354-C2D15D992494](https://i.loli.net/2019/04/01/5ca220b1981ad.png)
3. 将$C_1$中包含的核心对象从$Ω$中去除：$Ω=Ω\backslash C_1= \{\boldsymbol{x}_3, \boldsymbol{x}_5, \boldsymbol{x}_9, \boldsymbol{x}_{13}, \boldsymbol{x}_{14}, \boldsymbol{x}_{24}, \boldsymbol{x}_{25}, \boldsymbol{x}_{28}, \boldsymbol{x}_{29} \}$
4. 从更新后的集合$Ω$中随机选取一个核心对象作为种子，用于生成下一个聚类簇。
3. 不断重复上述过程，直至$Ω$为空。

9.10显示出DBSCAN先后生成聚类簇的情况。$C_1$之后生成的聚类簇为
![B9E0C13D-C12C-472E-97B1-F842D25D6089](https://i.loli.net/2019/04/01/5ca220b7de19a.png)
![D500E646-9263-4F25-A748-8271167D0CD2](https://i.loli.net/2019/04/01/5ca220a347590.png)


## 9.6 层次聚类 hierarchical clustering

试图在不同的层次对数据集进行划分，从而形成树形的聚类结构。数据集的划分可采用“自底向上”的聚合策略，也可采用“自顶向下”的拆分策略。

**代表算法：** AGNES（AGglomerative NESting）（自底向上）

**思路：**

先将数据集中的每个样本看做一个初始聚类簇，然后在算法运行的每一步中找出距离最近的两个聚类簇进行合并，不断重复上述过程直至达到预设的聚类簇个数。

**关键：**计算聚类簇间的距离

每个聚类簇是一个样本集合，采用关于集合的某种距离即可。例如，给定聚类簇$C_i$与$C_j$，可通过以下方式计算距离：
![-w602](https://i.loli.net/2019/04/01/5ca220c11d47d.jpg)

> 集合间的距离计算常采用豪斯多夫距离(Hausdorff distance)。

最小距离由两个簇的最近样本决定，最大距离由两个簇的最远样本决定，平均距离则由两个簇所有样本共同决定。当聚类簇距离由$d_{\min}$、$d_{\max}$或$d_{avg}$计算时，AGNES算法被相应地称为“单链接”(single-linkage)、“全链接”(complete-linkage)或“均链接”(average-linkage)算法。

算法描述：
![354DF491-4795-4F9C-95E3-AFD90D68621D](https://i.loli.net/2019/04/01/5ca220c7aaa61.png)

**说明：**

* 距离度量函数$d$：通常使用$d_{\min}$、$d_{\max}$或$d_{avg}$。
* 1-9：先对仅含一个样本的初始聚类簇和相应的距离矩阵进行初始化；
* 2：初始化单样本聚类簇；
* 6：初始化聚类簇距离矩阵；
* 11-23：不断合并距离最近的聚类簇，并对合并得到的聚类簇的距离矩阵进行更新；
* 12：i*<j*。
* 不断重复，直至达到预设的聚类数目。

**举一个例子🌰：**

以西瓜数据集4.0为例。令AGNES算法一直执行到所有样本出现在同一个簇中，即$k=1$，则可得到图9.12所示的“树状图”(dendrogram)，其中每层链接一组聚类簇。
![3F20CA93-B9A2-461A-AA88-1D09A0651A44](https://i.loli.net/2019/04/01/5ca220a0968b6.png)

在树状图的特定层次上进行分割，可得到相应的簇划分结果。例如，以图9.12中所示虚线分割树状图，将得到包含7个聚类簇的结果：
![2A0E5E78-125A-4457-B885-B092C15CA2A3](https://i.loli.net/2019/04/01/5ca220c18629a.png)

将分割层逐步上移，则可得到聚类簇逐渐减少的聚类结果。例如，图9.13显示了从图9.12中产生的7至4个聚类簇的划分结果。
![25C0F644-50C1-41F0-8505-72599BC32D](https://i.loli.net/2019/04/01/5ca220c7a25fe.png)

