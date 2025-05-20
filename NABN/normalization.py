import jittor as jt
from jittor import nn
from jittor import flatten, init, Module



class BatchNorm(Module):
    '''
    对输入进行批次归一化。
    '''

    def __init__(self, num_features, eps=1e-5, theta=0.2, momentum=0.1, affine=True, is_train=True, sync=True):
        # 是否在多节点之间同步均值与方差
        self.sync = sync
        # 特征维度（通道数）
        self.num_features = num_features
        # 是否处于训练模式（影响是否使用 running_mean/var）
        self.is_train = is_train
        # 防止除零的小常数
        self.eps = eps
        # 指数滑动平均的动量因子
        self.momentum = momentum
        # 是否使用 scale (gamma) 和 shift (beta) 参数
        self.affine = affine
        # 初始化可学习参数 weight（即 gamma），若 affine=False 则设为常数1
        self.weight = init.constant((num_features,), "float32", 1.0) if affine else 1.0
        # 初始化可学习参数 bias（即 beta），若 affine=False 则设为常数0
        self.bias = init.constant((num_features,), "float32", 0.0) if affine else 0.0
        # 初始化并停止梯度的运行时均值
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        # 初始化并停止梯度的运行时方差
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()

        # theta 不是 BatchNorm 标准参数，可能用于调试或扩展
        self.theta = theta
        print('self.theta is', self.theta)

    def execute(self, x):
        # 计算归一化维度（除了 batch 和 channel，剩下的都是）
        dims = [0]+list(range(2,x.ndim))
        
        if self.is_train:
            # 计算当前 batch 的均值
            xmean = jt.mean(x, dims=dims)
            # 计算当前 batch 的平方均值
            x2mean = jt.mean(x*x, dims=dims)

            # 若使用 MPI 分布式训练，则跨节点同步均值与平方均值
            if self.sync and jt.in_mpi:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            # 方差 = E[x²] - (E[x])²，maximum 用于防止负值
            xvar = (x2mean - xmean * xmean).maximum(0.0)

            # 计算归一化的 scale（w）和 shift（b）
            w = self.weight / jt.sqrt(xvar + self.eps)
            b = self.bias - xmean * w

            # 对输入进行归一化并加权偏移
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)

            # 更新 running_mean 和 running_var（指数滑动平均）
            self.running_mean.update(self.running_mean + (xmean.reshape((-1,)) - self.running_mean) * self.momentum)
            self.running_var.update(self.running_var + (xvar.reshape((-1,)) - self.running_var) * self.momentum)

            return norm_x
        else:
            # 测试时使用保存的 running_mean 和 running_var
            w = self.weight / jt.sqrt(self.running_var + self.eps)
            b = self.bias - self.running_mean * w

            # 使用保存的参数进行归一化
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
            return norm_x



class GauBatchNorm(Module):
    '''
    对输入进行批次归一化。
    '''

    def __init__(self, num_features, eps=1e-5, theta=0.2, momentum=0.1, affine=True, is_train=True, sync=True):
        # 是否在多节点之间同步均值与方差
        self.sync = sync
        # 特征维度（通道数）
        self.num_features = num_features
        # 是否处于训练模式（影响是否使用 running_mean/var）
        self.is_train = is_train
        # 防止除零的小常数
        self.eps = eps
        # 指数滑动平均的动量因子
        self.momentum = momentum
        # 是否使用 scale (gamma) 和 shift (beta) 参数
        self.affine = affine
        # 初始化可学习参数 weight（即 gamma），若 affine=False 则设为常数1
        self.weight = init.constant((num_features,), "float32", 1.0) if affine else 1.0
        # 初始化可学习参数 bias（即 beta），若 affine=False 则设为常数0
        self.bias = init.constant((num_features,), "float32", 0.0) if affine else 0.0
        # 初始化并停止梯度的运行时均值
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        # 初始化并停止梯度的运行时方差
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()

        # theta 不是 BatchNorm 标准参数，可能用于调试或扩展
        self.theta = theta
        print('self.theta is', self.theta)

    def execute(self, x):
        # 归一化维度：除掉batch和channel维度
        dims = [0] + list(range(2, x.ndim))
    
        if self.is_train:
            # 计算原始的 batch 均值 xmean
            xmean = jt.mean(x, dims=dims)  # shape: [C]
    
            # 从正态分布中采样：均值为 xmean，标准差为 1
            xmean_normal = jt.randn(xmean.shape) + xmean  # shape: [C]
    
            # 利用 mask 筛选出满足 [-theta+xmean, +theta+xmean] 范围的值
            lower = xmean - self.theta
            upper = xmean + self.theta
            mask = (xmean_normal >= lower) & (xmean_normal <= upper)
            # 用 mask 过滤：保留在区间内的值，其他用 xmean 代替
            xmean_filtered = jt.where(mask, xmean_normal, xmean)  # shape: [C]
    
            # 使用替代后的 xmean_filtered 计算 E[x²]
            x2mean = jt.mean(x * x, dims=dims)  # shape: [C]
    
            # 同步计算（若使用 MPI 多机训练）
            if self.sync and jt.in_mpi:
                xmean_filtered = xmean_filtered.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")
    
            # 用新均值计算新方差（确保非负）
            xvar = (x2mean - xmean_filtered * xmean_filtered).maximum(0.0)
    
            # 归一化比例因子：gamma / sqrt(var + eps)
            w = self.weight / jt.sqrt(xvar + self.eps)
            # 偏置项修正：beta - mu * scale
            b = self.bias - xmean_filtered * w
    
            # 应用归一化 + 仿射变换
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
    
            # 更新 running mean 和 var（使用新均值和方差）
            self.running_mean.update(self.running_mean + (xmean_filtered.reshape((-1,)) - self.running_mean) * self.momentum)
            self.running_var.update(self.running_var + (xvar.reshape((-1,)) - self.running_var) * self.momentum)
    
            return norm_x
    
        else:
            # 测试阶段使用保存的 running statistics
            w = self.weight / jt.sqrt(self.running_var + self.eps)
            b = self.bias - self.running_mean * w
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
            return norm_x






