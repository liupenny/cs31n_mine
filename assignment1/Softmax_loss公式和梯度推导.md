# Softmax loss公式和梯度推导

**写在前面：**

1. 这仅仅是自己的学习笔记，如果侵权，还请告知;
2. 讲义是参照[杜客](https://zhuanlan.zhihu.com/p/21930884)等人对cs231n的中文翻译。

## **1. 预备知识**

softmax求导：

首先给出数据形式：

![X：500\times3703](https://www.zhihu.com/equation?tex=X%EF%BC%9A500%5Ctimes3703) ；![W: 3703\times10](https://www.zhihu.com/equation?tex=W%3A+3703%5Ctimes10)

![X=[X_{1},…,X_{i},…,X_{N}]^{T}，i={\left\{ 1,2,…,N \right\}},N=500,X_{i}为1\times3703](https://www.zhihu.com/equation?tex=X%3D%5BX_%7B1%7D%2C%E2%80%A6%2CX_%7Bi%7D%2C%E2%80%A6%2CX_%7BN%7D%5D%5E%7BT%7D%EF%BC%8Ci%3D%7B%5Cleft%5C%7B+1%2C2%2C%E2%80%A6%2CN+%5Cright%5C%7D%7D%2CN%3D500%2CX_%7Bi%7D%E4%B8%BA1%5Ctimes3703)

![W=[W_{1},…,W_{j},…,W_{M}]，j={\left\{ 1,2,…,M \right\}},M=10,W_{j}为3703\times1](https://www.zhihu.com/equation?tex=W%3D%5BW_%7B1%7D%2C%E2%80%A6%2CW_%7Bj%7D%2C%E2%80%A6%2CW_%7BM%7D%5D%EF%BC%8Cj%3D%7B%5Cleft%5C%7B+1%2C2%2C%E2%80%A6%2CM+%5Cright%5C%7D%7D%2CM%3D10%2CW_%7Bj%7D%E4%B8%BA3703%5Ctimes1)

评分函数为：

![F=X.*W=\left[ F_{1},F_{2},…,F_{N}\right]^{T}](https://www.zhihu.com/equation?tex=F%3DX.%2AW%3D%5Cleft%5B+F_%7B1%7D%2CF_%7B2%7D%2C%E2%80%A6%2CF_%7BN%7D%5Cright%5D%5E%7BT%7D) ， 其中 ![F](https://www.zhihu.com/equation?tex=F) 为 ![500\times10](https://www.zhihu.com/equation?tex=500%5Ctimes10) ![F_{i}=X_{i}.*W=\left[ f_{1},…,f_{j},…,f_{M} \right]，F_{i}](https://www.zhihu.com/equation?tex=F_%7Bi%7D%3DX_%7Bi%7D.%2AW%3D%5Cleft%5B+f_%7B1%7D%2C%E2%80%A6%2Cf_%7Bj%7D%2C%E2%80%A6%2Cf_%7BM%7D+%5Cright%5D%EF%BC%8CF_%7Bi%7D) 为 ![1\times10](https://www.zhihu.com/equation?tex=1%5Ctimes10)

损失函数公式：

![L=\frac{1}{N}\sum_{i}^{}{L_{i}}+\lambda\ast[R\left( W \right)]](https://www.zhihu.com/equation?tex=L%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D%5E%7B%7D%7BL_%7Bi%7D%7D%2B%5Clambda%5Cast%5BR%5Cleft%28+W+%5Cright%29%5D)

\#######################################

若从考虑每个样本 ![i](https://www.zhihu.com/equation?tex=i) 的角度考虑，有

![L_{i}=-logp_{y_{i}}=-log\left( \frac{e^{f_{y_{i}}}}{\sum_{j}^{}{{e^{f_{j}}}}} \right)=-f_{y_{i}}+log{\sum_{j}^{}{{e^{f_{j}}}}}](https://www.zhihu.com/equation?tex=L_%7Bi%7D%3D-logp_%7By_%7Bi%7D%7D%3D-log%5Cleft%28+%5Cfrac%7Be%5E%7Bf_%7By_%7Bi%7D%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7B%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D%7D+%5Cright%29%3D-f_%7By_%7Bi%7D%7D%2Blog%7B%5Csum_%7Bj%7D%5E%7B%7D%7B%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D%7D) ，其中 ![y_{i}=\left\{ 1,2,…,M \right\}](https://www.zhihu.com/equation?tex=y_%7Bi%7D%3D%5Cleft%5C%7B+1%2C2%2C%E2%80%A6%2CM+%5Cright%5C%7D)

![\frac{dL}{dW} = \left[ \frac{∂L}{∂W_{1}}, \frac{∂L}{∂W_{2}}, …，\frac{∂L}{∂W_{M}}\right]](https://www.zhihu.com/equation?tex=%5Cfrac%7BdL%7D%7BdW%7D+%3D+%5Cleft%5B+%5Cfrac%7B%E2%88%82L%7D%7B%E2%88%82W_%7B1%7D%7D%2C+%5Cfrac%7B%E2%88%82L%7D%7B%E2%88%82W_%7B2%7D%7D%2C+%E2%80%A6%EF%BC%8C%5Cfrac%7B%E2%88%82L%7D%7B%E2%88%82W_%7BM%7D%7D%5Cright%5D)

因为 ![L=\frac{1}{N}\sum_{i}^{}{L_{i}}+\lambda\ast[R\left( W \right)]](https://www.zhihu.com/equation?tex=L%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D%5E%7B%7D%7BL_%7Bi%7D%7D%2B%5Clambda%5Cast%5BR%5Cleft%28+W+%5Cright%29%5D)

所以求 ![\frac{dL}{dW}](https://www.zhihu.com/equation?tex=%5Cfrac%7BdL%7D%7BdW%7D) 相当于求 ![\frac{dL_{i}}{dW}=\left[ \frac{∂L_{i}}{∂W_{1}}, \frac{∂L_{i}}{∂W_{2}}, …，\frac{∂L_{i}}{∂W_{M}}\right]](https://www.zhihu.com/equation?tex=%5Cfrac%7BdL_%7Bi%7D%7D%7BdW%7D%3D%5Cleft%5B+%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W_%7B1%7D%7D%2C+%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W_%7B2%7D%7D%2C+%E2%80%A6%EF%BC%8C%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W_%7BM%7D%7D%5Cright%5D)

所以只要求 ![\frac{∂L_{i}}{∂W_{j}}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W_%7Bj%7D%7D) 即可，其中 ![j=\left\{ 1,2,…,M \right\}](https://www.zhihu.com/equation?tex=j%3D%5Cleft%5C%7B+1%2C2%2C%E2%80%A6%2CM+%5Cright%5C%7D)

![\frac{∂L_{i}}{∂W_{j}}=\frac{∂}{∂W_{j}}\left( -f_{y_{i}}+log{\sum_{j}^{}{{e^{f_{j}}}}} \right) =\frac{∂\left( -f_{y_{i}} \right)}{∂W_{j}}+\frac{∂\left( log{\sum_{j}^{}{{e^{f_{j}}}}} \right)}{∂W_{j}}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W_%7Bj%7D%7D%3D%5Cfrac%7B%E2%88%82%7D%7B%E2%88%82W_%7Bj%7D%7D%5Cleft%28+-f_%7By_%7Bi%7D%7D%2Blog%7B%5Csum_%7Bj%7D%5E%7B%7D%7B%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D%7D+%5Cright%29+%3D%5Cfrac%7B%E2%88%82%5Cleft%28+-f_%7By_%7Bi%7D%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D%2B%5Cfrac%7B%E2%88%82%5Cleft%28+log%7B%5Csum_%7Bj%7D%5E%7B%7D%7B%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D)

**首先求** ![\frac{∂\left( -f_{y_{i}} \right)}{∂W_{j}}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82%5Cleft%28+-f_%7By_%7Bi%7D%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D) ：

当 ![j==y_{i}](https://www.zhihu.com/equation?tex=j%3D%3Dy_%7Bi%7D) 时， ![\frac{∂\left( -f_{y_{i}} \right)}{∂W_{j}}=-\frac{∂\left( X_{i,j}.*W_{j} \right)}{∂W_{j}}=-X_{i,j}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82%5Cleft%28+-f_%7By_%7Bi%7D%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D%3D-%5Cfrac%7B%E2%88%82%5Cleft%28+X_%7Bi%2Cj%7D.%2AW_%7Bj%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D%3D-X_%7Bi%2Cj%7D)

当 ![j!=y_{i}](https://www.zhihu.com/equation?tex=j%21%3Dy_%7Bi%7D) 时， ![\frac{∂\left( -f_{y_{i}} \right)}{∂W_{j}}=-\frac{∂\left( X_{y_{i},j}.*W_{y_{i}} \right)}{∂W_{j}}=0](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82%5Cleft%28+-f_%7By_%7Bi%7D%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D%3D-%5Cfrac%7B%E2%88%82%5Cleft%28+X_%7By_%7Bi%7D%2Cj%7D.%2AW_%7By_%7Bi%7D%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D%3D0)

**其次求** ![\frac{∂\left( log{\sum_{j}^{}{{e^{f_{j}}}}} \right)}{∂W_{j}}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82%5Cleft%28+log%7B%5Csum_%7Bj%7D%5E%7B%7D%7B%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D) ：

![\frac{∂\left( log{\sum_{j}^{}{{e^{f_{j}}}}} \right)}{∂W_{j}}=\frac{1}{\sum_{j}^{}{e^{f_{j}}}} \frac{∂{\sum_{j}^{}{e^{f_{j}}}}}{∂W_{j}} =\frac{1}{\sum_{j}^{}{e^{f_{j}}}} \sum_{j}^{}{\frac{∂e^{f_{j}}}{∂W_{j}}} ](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82%5Cleft%28+log%7B%5Csum_%7Bj%7D%5E%7B%7D%7B%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D%3D%5Cfrac%7B1%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D+%5Cfrac%7B%E2%88%82%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D%7D%7B%E2%88%82W_%7Bj%7D%7D+%3D%5Cfrac%7B1%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D+%5Csum_%7Bj%7D%5E%7B%7D%7B%5Cfrac%7B%E2%88%82e%5E%7Bf_%7Bj%7D%7D%7D%7B%E2%88%82W_%7Bj%7D%7D%7D+)

![=\frac{e^{f_{j}}}{\sum_{j}^{}{e^{f_{j}}}} \sum_{j}^{}{\frac{∂\left(f_{j} \right)}{∂W_{j}}} =\frac{e^{f_{j}}}{\sum_{j}^{}{e^{f_{j}}}} \sum_{j}^{}{\frac{∂\left( X_{i,j}.*W_{j} \right)}{∂W_{j}}} =\frac{e^{f_{j}}}{\sum_{j}^{}{e^{f_{j}}}}X_{i,j}](https://www.zhihu.com/equation?tex=%3D%5Cfrac%7Be%5E%7Bf_%7Bj%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D+%5Csum_%7Bj%7D%5E%7B%7D%7B%5Cfrac%7B%E2%88%82%5Cleft%28f_%7Bj%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D%7D+%3D%5Cfrac%7Be%5E%7Bf_%7Bj%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D+%5Csum_%7Bj%7D%5E%7B%7D%7B%5Cfrac%7B%E2%88%82%5Cleft%28+X_%7Bi%2Cj%7D.%2AW_%7Bj%7D+%5Cright%29%7D%7B%E2%88%82W_%7Bj%7D%7D%7D+%3D%5Cfrac%7Be%5E%7Bf_%7Bj%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7DX_%7Bi%2Cj%7D)

**最终：**

当 ![j==y_{i}](https://www.zhihu.com/equation?tex=j%3D%3Dy_%7Bi%7D) 时， ![\frac{∂L_{i}}{∂W_{j}}=\frac{e^{f_{j}}}{\sum_{j}^{}{e^{f_{j}}}}X_{i,j}-X_{i,j}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W_%7Bj%7D%7D%3D%5Cfrac%7Be%5E%7Bf_%7Bj%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7DX_%7Bi%2Cj%7D-X_%7Bi%2Cj%7D)

当 ![j!=y_{i}](https://www.zhihu.com/equation?tex=j%21%3Dy_%7Bi%7D) 时， ![\frac{∂L_{i}}{∂W_{j}}=\frac{e^{f_{j}}}{\sum_{j}^{}{e^{f_{j}}}}X_{i,j}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W_%7Bj%7D%7D%3D%5Cfrac%7Be%5E%7Bf_%7Bj%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7DX_%7Bi%2Cj%7D)

\#######################################

若从样本整体 ![X](https://www.zhihu.com/equation?tex=X) 的角度来理解，则有：

![\frac{dL}{dW}=\frac{d}{dW} \left( \frac{1}{N}\sum_{i}^{}{L_{i}}+\lambda\ast[R\left( W \right)] \right)](https://www.zhihu.com/equation?tex=%5Cfrac%7BdL%7D%7BdW%7D%3D%5Cfrac%7Bd%7D%7BdW%7D+%5Cleft%28+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D%5E%7B%7D%7BL_%7Bi%7D%7D%2B%5Clambda%5Cast%5BR%5Cleft%28+W+%5Cright%29%5D+%5Cright%29)

**只考虑前半部分（后半部分简单）：**

![pre\frac{dL}{dW}=\frac{d}{dW} \left( \frac{1}{N}\sum_{i}^{}{L_{i}} \right)](https://www.zhihu.com/equation?tex=pre%5Cfrac%7BdL%7D%7BdW%7D%3D%5Cfrac%7Bd%7D%7BdW%7D+%5Cleft%28+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D%5E%7B%7D%7BL_%7Bi%7D%7D+%5Cright%29)

令 ![L=\left[ L_{1},…,L_{i},…,L_{N} \right]](https://www.zhihu.com/equation?tex=L%3D%5Cleft%5B+L_%7B1%7D%2C%E2%80%A6%2CL_%7Bi%7D%2C%E2%80%A6%2CL_%7BN%7D+%5Cright%5D) ,则有 ![\frac{dL}{dW}=\left[ \frac{∂L_{1}}{∂W},…,\frac{∂L_{i}}{∂W},…,\frac{∂L_{N}}{∂W} \right]](https://www.zhihu.com/equation?tex=%5Cfrac%7BdL%7D%7BdW%7D%3D%5Cleft%5B+%5Cfrac%7B%E2%88%82L_%7B1%7D%7D%7B%E2%88%82W%7D%2C%E2%80%A6%2C%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W%7D%2C%E2%80%A6%2C%5Cfrac%7B%E2%88%82L_%7BN%7D%7D%7B%E2%88%82W%7D+%5Cright%5D)

而![\frac{∂L_{i}}{∂W}=\frac{∂L_{i}}{∂F}\frac{∂F}{∂W}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W%7D%3D%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F%7D%5Cfrac%7B%E2%88%82F%7D%7B%E2%88%82W%7D)

**step1，求** ![\frac{∂L_{i}}{∂F}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F%7D) ：

因为： ![F=X.*W=\left[ F_{1},F_{2},…,F_{N}\right]^{T}](https://www.zhihu.com/equation?tex=F%3DX.%2AW%3D%5Cleft%5B+F_%7B1%7D%2CF_%7B2%7D%2C%E2%80%A6%2CF_%7BN%7D%5Cright%5D%5E%7BT%7D) ，所以

![\frac{∂L_{i}}{∂F}=\left[ \frac{∂L_{i}}{∂F_{1}},…,\frac{∂L_{i}}{∂F_{i}},…,\frac{∂L_{i}}{∂F_{N}} \right]^{T}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F%7D%3D%5Cleft%5B+%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F_%7B1%7D%7D%2C%E2%80%A6%2C%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F_%7Bi%7D%7D%2C%E2%80%A6%2C%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F_%7BN%7D%7D+%5Cright%5D%5E%7BT%7D)

因为： ![F_{i}=X_{i}.*W=\left[ f_{1},…,f_{j},…,f_{M} \right]，F_{i}](https://www.zhihu.com/equation?tex=F_%7Bi%7D%3DX_%7Bi%7D.%2AW%3D%5Cleft%5B+f_%7B1%7D%2C%E2%80%A6%2Cf_%7Bj%7D%2C%E2%80%A6%2Cf_%7BM%7D+%5Cright%5D%EF%BC%8CF_%7Bi%7D) 为 ![1\times10](https://www.zhihu.com/equation?tex=1%5Ctimes10) ，所以 ![\frac{∂L_{i}}{∂F_{i}}=\left[ \frac{∂L_{i}}{∂f_{1}},…,\frac{∂L_{i}}{∂f_{i}},…,\frac{∂L_{i}}{∂f_{M}} \right]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F_%7Bi%7D%7D%3D%5Cleft%5B+%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82f_%7B1%7D%7D%2C%E2%80%A6%2C%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82f_%7Bi%7D%7D%2C%E2%80%A6%2C%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82f_%7BM%7D%7D+%5Cright%5D) ,而 ![\frac{∂L_{i}}{∂W_{j}}=\frac{∂L_{i}}{∂f_{i}}\frac{∂f_{i}}{∂W_{j}}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W_%7Bj%7D%7D%3D%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82f_%7Bi%7D%7D%5Cfrac%7B%E2%88%82f_%7Bi%7D%7D%7B%E2%88%82W_%7Bj%7D%7D) , 而![\frac{∂f_{i}}{∂W_{j}}=X_{i,j}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82f_%7Bi%7D%7D%7B%E2%88%82W_%7Bj%7D%7D%3DX_%7Bi%2Cj%7D)，根据上文可知，

当 ![j==y_{i}](https://www.zhihu.com/equation?tex=j%3D%3Dy_%7Bi%7D) 时， ![\frac{∂L_{i}}{∂W_{j}}=\frac{e^{f_{j}}}{\sum_{j}^{}{e^{f_{j}}}}X_{i,j}-X_{i,j}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W_%7Bj%7D%7D%3D%5Cfrac%7Be%5E%7Bf_%7Bj%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7DX_%7Bi%2Cj%7D-X_%7Bi%2Cj%7D)

当 ![j!=y_{i}](https://www.zhihu.com/equation?tex=j%21%3Dy_%7Bi%7D) 时， ![\frac{∂L_{i}}{∂W_{j}}=\frac{e^{f_{j}}}{\sum_{j}^{}{e^{f_{j}}}}X_{i,j}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W_%7Bj%7D%7D%3D%5Cfrac%7Be%5E%7Bf_%7Bj%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7DX_%7Bi%2Cj%7D)

所以

当 ![j==y_{i}](https://www.zhihu.com/equation?tex=j%3D%3Dy_%7Bi%7D) 时， ![\frac{∂L_{i}}{∂f_{i}}=\frac{e^{f_{j}}}{\sum_{j}^{}{e^{f_{j}}}}-1](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82f_%7Bi%7D%7D%3D%5Cfrac%7Be%5E%7Bf_%7Bj%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D-1)

当 ![j!=y_{i}](https://www.zhihu.com/equation?tex=j%21%3Dy_%7Bi%7D) 时， ![\frac{∂L_{i}}{∂f_{i}}=\frac{e^{f_{j}}}{\sum_{j}^{}{e^{f_{j}}}}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82f_%7Bi%7D%7D%3D%5Cfrac%7Be%5E%7Bf_%7Bj%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D)

![\frac{∂L_{i}}{∂F_{i}}=\left[ \frac{∂L_{i}}{∂f_{1}},…,\frac{∂L_{i}}{∂f_{i}},…,\frac{∂L_{i}}{∂f_{M}} \right] =\frac{e^{f_{j}}}{\sum_{j}^{}{e^{f_{j}}}}-\left[ 0,…,1,…,0 \right]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F_%7Bi%7D%7D%3D%5Cleft%5B+%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82f_%7B1%7D%7D%2C%E2%80%A6%2C%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82f_%7Bi%7D%7D%2C%E2%80%A6%2C%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82f_%7BM%7D%7D+%5Cright%5D+%3D%5Cfrac%7Be%5E%7Bf_%7Bj%7D%7D%7D%7B%5Csum_%7Bj%7D%5E%7B%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D-%5Cleft%5B+0%2C%E2%80%A6%2C1%2C%E2%80%A6%2C0+%5Cright%5D) ,其中 ![1](https://www.zhihu.com/equation?tex=1)为当 ![j==y_{i}](https://www.zhihu.com/equation?tex=j%3D%3Dy_%7Bi%7D) 时的取值,所以 ![\frac{∂L_{i}}{∂F_{i}}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F_%7Bi%7D%7D) 为 ![1\times10](https://www.zhihu.com/equation?tex=1%5Ctimes10)

所以 ![\frac{∂L_{i}}{∂F}=\left[ \frac{∂L_{i}}{∂F_{1}},…,\frac{∂L_{i}}{∂F_{i}},…,\frac{∂L_{i}}{∂F_{N}} \right]^{T}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F%7D%3D%5Cleft%5B+%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F_%7B1%7D%7D%2C%E2%80%A6%2C%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F_%7Bi%7D%7D%2C%E2%80%A6%2C%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F_%7BN%7D%7D+%5Cright%5D%5E%7BT%7D) 为 ![500\times10](https://www.zhihu.com/equation?tex=500%5Ctimes10)

**step2，求** ![\frac{∂F}{∂W}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82F%7D%7B%E2%88%82W%7D) ：

因为 ![\frac{∂L_{i}}{∂W}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W%7D) 为 ![3703\times10](https://www.zhihu.com/equation?tex=3703%5Ctimes10) ，而 ![\frac{∂L_{i}}{∂F}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F%7D) 为 ![500\times10](https://www.zhihu.com/equation?tex=500%5Ctimes10) ，所以， ![\frac{∂F}{∂W}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82F%7D%7B%E2%88%82W%7D) 为 ![500\times3073](https://www.zhihu.com/equation?tex=500%5Ctimes3073)

**疑问：**

**![\frac{∂L_{i}}{∂W}=\frac{∂F}{∂W}\frac{∂L_{i}}{∂F}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W%7D%3D%5Cfrac%7B%E2%88%82F%7D%7B%E2%88%82W%7D%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F%7D) 等价于 ![\frac{∂L_{i}}{∂W}=\frac{∂L_{i}}{∂F}\frac{∂F}{∂W}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W%7D%3D%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F%7D%5Cfrac%7B%E2%88%82F%7D%7B%E2%88%82W%7D) ，是求** ![\frac{∂L_{i}}{∂F}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F%7D) 与 ![\frac{∂F}{∂W}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82F%7D%7B%E2%88%82W%7D) 的内积，根据内积公式及dW的维度可知，应采用 ![\frac{∂L_{i}}{∂W}=\frac{∂F}{∂W}\frac{∂L_{i}}{∂F}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82W%7D%3D%5Cfrac%7B%E2%88%82F%7D%7B%E2%88%82W%7D%5Cfrac%7B%E2%88%82L_%7Bi%7D%7D%7B%E2%88%82F%7D) 进行推导。

