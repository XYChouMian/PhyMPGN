# PyG Transform 详解

## 📋 文档概述

本文档详细解析 PhyMPGN 项目中的 [transform.py](./cylinder_flow/src/datasets/transform.py) 文件，该文件定义了用于图数据处理的各类变换（Transform），实现从原始CFD数据到图神经网络输入格式的完整转换流程。

---

## 🔍 一、Transform 基础概念

### 1.1 什么是 Transform？

在 PyTorch Geometric 中，**Transform 是一个对图数据执行操作的函数对象**。

```python
# Transform 的基本使用方式
transform = SomeTransform()
new_data = transform(old_data)  # 输入图数据，返回修改后的图数据
```

**类比理解**：
- **图像处理**：`torchvision.transforms` 对图片进行旋转、裁剪、归一化
- **图处理**：`torch_geometric.transforms` 对图进行添加特征、构建拓扑、标记节点

---

### 1.2 BaseTransform 抽象基类

`BaseTransform` 是所有变换的抽象基类，定义了统一接口。

#### 核心结构

```python
from torch_geometric.transforms import BaseTransform

class BaseTransform:
    """所有图变换的基类"""
    
    def __call__(self, data):
        """
        核心方法：接收 Data 对象，返回修改后的 Data 对象
        
        Args:
            data (Data): 输入的图数据
            
        Returns:
            Data: 修改后的图数据
        """
        raise NotImplementedError  # 子类必须实现
    
    def __repr__(self):
        """返回变换的字符串表示"""
        return f'{self.__class__.__name__}()'
```

#### 关键特点

1. **必须实现 `__call__` 方法**：定义具体的变换逻辑
2. **支持对象携带状态**：通过 `__init__` 接收参数并保存
3. **接口统一**：所有Transform都遵循 `transform(data)` 的调用方式

---

### 1.3 自定义 Transform 的标准模式

```python
from torch_geometric.transforms import BaseTransform
import torch

class MyCustomTransform(BaseTransform):
    """自定义变换示例"""
    
    # 1. 初始化参数（可选）
    def __init__(self, parameter=True):
        self.parameter = parameter
    
    # 2. 实现核心逻辑（必须）
    def __call__(self, data):
        # 对数据进行修改
        data.new_attribute = some_value
        return data  # 必须返回 Data 对象
    
    # 3. 字符串表示（可选）
    def __repr__(self):
        return f'MyCustomTransform(parameter={self.parameter})'
```

---

## 🎯 二、函数式装饰器机制

### 2.1 @functional_transform 装饰器

PyG 提供的语法糖，让 Transform 支持两种调用方式。

#### 装饰器作用

```python
from torch_geometric.data.datapipes import functional_transform

@functional_transform('my_custom')
class MyCustomTransform(BaseTransform):
    def __init__(self, param1=False, param2=1.0):  # 支持参数
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, data):
        # 变换逻辑
        return data
```

**等价效果**：

```python
# 方式1：类实例化（标准方式，推荐）
transform = MyCustomTransform(param1=True, param2=2.0)
data = transform(data)

# 方式2：函数式调用（装饰器提供，同样支持传参）
import torch_geometric.transforms as T
data = T.my_custom(data, param1=True, param2=2.0)  # 参数直接传给 __init__
```

#### 装饰器原理

装饰器自动创建包装函数处理参数传递：

```python
# 装饰器等价于自动注册：
T.my_custom = lambda data, **kwargs: MyCustomTransform(**kwargs)(data)
# 先实例化类 → 再调用 __call__
```

**关键点**：
- 装饰器参数 `'my_custom'` 是注册的函数名（通常用小写+下划线）
- 函数式调用内部仍会创建类实例，只是语法更简洁
- **两种方式都支持参数传递**，传给 `__init__` 方法
- 实际项目中类实例化方式更常见（IDE 支持更好，类型提示更清晰）

**优势**：
- 代码更简洁
- 与 PyG 内置变换保持一致的调用风格
- 便于在 `T.Compose` 中使用

---

### 2.2 Transform 组合机制

#### T.Compose 变换链

```python
from torch_geometric.transforms import Compose

transform = Compose([
    MyDistance(norm=True),      # 第一步：计算边长
    MyCartesian(norm=True),     # 第二步：计算方向向量
])

data = transform(data)  # 按顺序执行所有变换
```

#### 执行流程

```python
# 逐个应用变换的等价代码
data = MyDistance(norm=True)(data)    # 先执行
data = MyCartesian(norm=True)(data)   # 再执行
```

#### 在数据集中的使用

```python
class MyDataset(InMemoryDataset):
    def __init__(self, root, pre_transform=None):
        super().__init__(root, pre_transform=pre_transform)

# pre_transform：预处理时执行一次
pre_transform = Compose([
    Delaunay(),           # 构建三角网格
    MyDistance(norm=True) # 计算边长
])

dataset = MyDataset(root='data/', pre_transform=pre_transform)
# 第一次运行：应用变换并缓存到 processed/data.pt
# 后续运行：直接加载缓存，跳过变换
```

---

## 🔧 三、核心 Transform 类详解

### 3.1 几何特征变换

#### MyDistance — 边长度计算

**文件位置**: [transform.py#L49-80](./cylinder_flow/src/datasets/transform.py#L49-80)

```python
@functional_transform('my_distance')
class MyDistance(BaseTransform):
    def __init__(self, norm: bool = False):
        self.norm = norm  # 是否归一化
    
    def __call__(self, data: Data) -> Data:
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        
        # 计算欧几里得距离
        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        data.distance = dist
        
        # 归一化处理
        if self.norm and dist.numel() > 0:
            dist = dist / dist.max()
        
        # 拼接到边特征
        if pseudo is not None:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist
        
        return data
```

**功能说明**：
1. 计算每条边的欧几里得长度
2. 可选归一化到 `[0, 1]` 范围
3. 将结果添加到边特征中

**使用示例**：

```python
# 输入：边 (节点0 → 节点1)
pos[0] = [1.0, 2.0]
pos[1] = [3.0, 5.0]
edge_index = [[0, 1], [1, 0]]  # 双向边

# 输出
dist = sqrt((3-1)² + (5-2)²) = sqrt(13) = 3.606
data.edge_attr = [[3.606], [3.606]]  # 两条边的长度相同
```

---

#### MyCartesian — 边方向向量计算

**文件位置**: [transform.py#L9-46](./cylinder_flow/src/datasets/transform.py#L9-46)

```python
@functional_transform('my_cartesian')
class MyCartesian(BaseTransform):
    def __init__(self, norm: bool = False):
        self.norm = norm
    
    def __call__(self, data: Data) -> Data:
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        
        # 计算相对位置向量
        cart = pos[col] - pos[row]  # 目标 - 源
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart
        data.rel_pos = cart
        
        # 特殊归一化到 [0, 1]
        if self.norm and cart.numel() > 0:
            max_value = cart.abs().max()
            cart = cart / (2 * max_value) + 0.5
        
        # 拼接到边特征
        if pseudo is not None:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart
        
        return data
```

**归一化原理**：
```
原始范围: [-max, +max]
除以 2*max: [-0.5, +0.5]
加 0.5: [0, 1]
```

**使用示例**：

```python
# 边 (节点0 → 节点1)
pos[0] = [1.0, 2.0]
pos[1] = [3.0, 5.0]

# 计算
cart = [3.0-1.0, 5.0-2.0] = [2.0, 3.0]  # 向右上方的向量

# 归一化（假设 max = 3.0）
cart_norm = [2.0/6.0 + 0.5, 3.0/6.0 + 0.5] = [0.833, 1.0]
```

---

### 3.2 边界条件变换

#### Dirichlet — 固定值边界

**文件位置**: [transform.py#L139-149](./cylinder_flow/src/datasets/transform.py#L139-149)

```python
@functional_transform('dirichlet')
class Dirichlet(BaseTransform):
    def __init__(self):
        self.index = None
    
    def set_index(self, index):
        self.index = torch.tensor(index, dtype=torch.long)
    
    def __call__(self, data):
        data.dirichlet_index = self.index  # 标记边界节点
        return data
```

**物理背景**：圆柱壁面无滑移条件（$u = 0$）

**使用方式**：

```python
# 在 PDEGraphDataset 中
dirichlet_trans = Dirichlet()
dirichlet_trans.set_index(cylinder_index)  # 圆柱壁面节点

# 应用变换
data = dirichlet_trans(data)
# data.dirichlet_index: 圆柱壁面节点的索引数组

# 设置边界值
data.dirichlet_value = torch.zeros((len(cylinder_index), 2))  # u=0, v=0
```

---

#### DirichletInlet — 入口边界

**文件位置**: [transform.py#L137-145](./cylinder_flow/src/datasets/transform.py#L137-145)

```python
@functional_transform('dirichlet_inlet')
class DirichletInlet(BaseTransform):
    def __init__(self):
        self.index = None
    
    def set_index(self, index):
        self.index = torch.tensor(index, dtype=torch.long)
    
    def __call__(self, data):
        data.inlet_index = self.index
        return data
```

**物理背景**：入口固定速度条件（$u = u_{in}$）

**使用方式**：

```python
# 标记入口节点
inlet_trans = DirichletInlet()
inlet_trans.set_index(inlet_index)

# 设置入口速度值
data.inlet_value = self.inlet_velocity(data.inlet_index, 1.0)
# u = 1.0 (归一化后的入口速度), v = 0
```

---

#### Periodic — 周期性边界（幽灵节点）

**文件位置**: [transform.py#L83-136](./cylinder_flow/src/datasets/transform.py#L83-136)

**物理背景**：模拟周期性流动（如大气环流），边界处节点在空间上是连续的。

**实现原理**：

```python
class Periodic(BaseTransform):
    def __init__(self, distance):
        self.y_min = 0.
        self.y_max = 8.
        self.distance = distance  # 幽灵节点距离
        self.ghost_pos = None
        self.source_index = None  # 真实节点索引
        self.target_index = None  # 幽灵节点索引
    
    def get_periodic_index(self, data):
        pos = data.pos
        ghost_pos = []
        tgt2src = {}
        target_count = pos.shape[0]
        
        for i in range(pos.shape[0]):
            x, y = pos[i]
            
            # 如果节点靠近下边界，在上边界对应位置创建幽灵节点
            if 0 < y - self.y_min <= self.distance:
                ghost_x = x
                ghost_y = y + (self.y_max - self.y_min)  # 周期性平移
                ghost_pos.append([ghost_x, ghost_y])
                tgt2src[target_count] = i
                target_count += 1
            
            # 如果节点靠近上边界，在下边界对应位置创建幽灵节点
            elif 0 < self.y_max - y <= self.distance:
                ghost_x = x
                ghost_y = y - (self.y_max - self.y_min)
                ghost_pos.append([ghost_x, ghost_y])
                tgt2src[target_count] = i
                target_count += 1
        
        self.ghost_pos = torch.tensor(ghost_pos)
        self.source_index = torch.tensor(list(tgt2src.values()))
        self.target_index = torch.tensor(list(tgt2src.keys()))
    
    def __call__(self, data):
        if self.is_none():
            self.get_periodic_index(data)
        
        # 保存映射关系
        data.periodic_src_index = self.source_index
        data.periodic_tgt_index = self.target_index
        
        # 添加幽灵节点
        data.pos = torch.cat((data.pos, self.ghost_pos), dim=0)
        ghost_y = torch.zeros((self.ghost_pos.shape[0], data.y.shape[1], data.y.shape[2]))
        data.y = torch.cat((data.y, ghost_y), dim=0)
        
        return data
```

**可视化效果**：

```
原始域（y ∈ [0, 8]）:
  上边界 y=8:  ●─────●─────●
               │     │     │
  下边界 y=0:  ○─────○─────○

添加幽灵节点后:
  幽灵(上) y=8.x:  ◎─────◎─────◎  ← 复制自下边界
  上边界   y=8:    ●─────●─────●
                   │     │     │
  下边界   y=0:    ○─────○─────○
  幽灵(下) y=-0.x: ◎─────◎─────◎  ← 复制自上边界
```

---

#### Neumann — 零梯度边界（幽灵节点）

**文件位置**: [transform.py#L165-266](./cylinder_flow/src/datasets/transform.py#L165-266)

**物理背景**：出口边界，速度梯度为零（$\frac{\partial u}{\partial n} = 0$）

**实现原理**：在边界外侧创建对称的幽灵节点

```python
class Neumann(BaseTransform):
    def __init__(self, distance, distance_circle):
        self.x0, self.y0 = 0.5, 0.3  # 圆柱中心
        self.r = 0.2                  # 圆柱半径
        self.bdry = 0.5               # 右边界位置
        self.distance = distance
        self.distance_circle = distance_circle
    
    def get_neumann_index(self, data):
        pos = data.pos
        ghost_pos = []
        tgt2src = {}
        target_count = pos.shape[0]
        
        # 直线边界对称（右边界）
        for i in range(pos.shape[0]):
            x1, y1 = pos[i]
            # 如果节点靠近右边界，在右边界外侧创建对称幽灵节点
            if 0 < self.bdry - x1 < self.distance and not on_circle(x1, y1):
                ghost_x = x1 + (self.bdry - x1) * 2  # 镜像对称
                ghost_y = y1
                ghost_pos.append([ghost_x, ghost_y])
                tgt2src[target_count] = i
                target_count += 1
        
        # 圆形边界对称（圆柱）
        for i in range(pos.shape[0]):
            x1, y1 = pos[i]
            if near_circle(x1, y1, self.distance_circle):
                # 沿法向量方向对称创建幽灵节点
                tan = (y1 - self.y0) / (x1 - self.x0)
                cos_2 = 1 / (1 + tan**2)
                sin_2 = tan**2 / (1 + tan**2)
                
                # 计算法向量方向
                cos = sqrt(cos_2) if x1 > self.x0 else -sqrt(cos_2)
                sin = sqrt(sin_2) if y1 > self.y0 else -sqrt(sin_2)
                
                # 对称位置
                ghost_x = 2 * (self.r * cos + self.x0) - x1
                ghost_y = 2 * (self.r * sin + self.y0) - y1
                
                ghost_pos.append([ghost_x, ghost_y])
                tgt2src[target_count] = i
                target_count += 1
        
        self.ghost_pos = torch.tensor(ghost_pos)
        self.source_index = torch.tensor(list(tgt2src.values()))
        self.target_index = torch.tensor(list(tgt2src.keys()))
```

**可视化效果**：

```
直线边界对称（右边界）:
边界 x=0.5:       幽灵区域
    │
  ● │            ◎   ← 幽灵节点（x=0.5 + ε）
  ● │            ◎
    │  ↑ 对称

圆形边界对称（圆柱）:
      ◎
    ◎   ◎
  ◎  ◎  ◎  ← 幽灵节点（外侧对称）
  ●  ●  ●  ← 圆柱壁面
    ◎  ◎
```

---

### 3.3 网格处理变换

#### MaskFace — 移除障碍物内部三角形

**文件位置**: [transform.py#L270-297](./cylinder_flow/src/datasets/transform.py#L270-297)

```python
@functional_transform('mask_face')
class MaskFace(BaseTransform):
    def __init__(self):
        self.cylinder_index = None
        self.new_face_index = None
    
    def set_cylinder_index(self, cylinder_index):
        self.cylinder_index = torch.tensor(cylinder_index, dtype=torch.long)
    
    def cal_mask_face(self, graph):
        on_circle_index = self.cylinder_index
        new_face_index = []
        
        for i in range(graph.face.shape[1]):
            # 如果三角形的三个顶点都在圆柱上，则移除
            if torch.isin(graph.face[:, i], on_circle_index).all():
                continue  # 跳过这个三角形
            else:
                new_face_index.append(i)
        
        return torch.tensor(new_face_index)
    
    def __call__(self, data):
        if self.is_none():
            self.new_face_index = self.cal_mask_face(data)
        
        data.face = data.face[:, self.new_face_index]
        return data
```

**功能说明**：
1. 检查每个三角形的所有顶点
2. 如果三个顶点都在障碍物（圆柱）上，则移除该三角形
3. 避免在障碍物内部生成无效的连接

**可视化效果**：

```
Delaunay 生成的三角形:    移除圆柱内部后:
   ◢───◣                    ◢───◣
  ◢─●─◣       →            ◢─   ─◣  (移除包含三个●的三角形)
   ◢───◣                    ◢───◣
```

---

#### NodeTypeInfo — 节点类型标记

**文件位置**: [transform.py#L300-330](./cylinder_flow/src/datasets/transform.py#L300-330)

```python
@functional_transform('node_type_info')
class NodeTypeInfo(BaseTransform):
    def __init__(self):
        self.type_dict = None
        self.node_type = None
    
    def set_type_dict(self, type_dict):
        self.type_dict = type_dict
    
    def cal_node_type(self, data):
        node_num = data.pos.shape[0]
        # 默认为内部节点
        node_type = torch.ones(node_num, dtype=torch.long) * NodeType.NORMAL
        
        # 标记圆柱壁面
        if hasattr(data, 'dirichlet_index'):
            node_type[data.dirichlet_index] = NodeType.OBSTACLE
        
        # 标记入口
        if hasattr(data, 'inlet_index'):
            node_type[data.inlet_index] = NodeType.INLET
        
        # 标记出口
        outlet_index = self.type_dict['outlet'][:]
        outlet_index = torch.tensor(outlet_index, dtype=torch.long)
        node_type[outlet_index] = NodeType.OUTLET
        
        return node_type
    
    def __call__(self, data):
        if self.is_none():
            self.node_type = self.cal_node_type(data)
        
        data.node_type = self.node_type
        return data
```

**节点类型枚举**：

```python
class NodeType:
    NORMAL = 0     # 内部流动节点
    OBSTACLE = 1   # 圆柱壁面节点
    INLET = 2      # 入口节点
    OUTLET = 3     # 出口节点
```

**用途**：
1. 作为节点特征输入模型（one-hot编码）
2. 指导边界条件的应用逻辑
3. 用于可视化和分析

---

## 🔄 四、Transform 组合与执行流程

### 4.1 在 PDEGraphDataset 中的应用

**文件位置**: [dataset.py#L34-64](./cylinder_flow/src/datasets/dataset.py#L34-64)

```python
class PDEGraphDataset(InMemoryDataset):
    def __init__(self, ...):
        ...

        # 1. 创建独立的变换对象
        self.periodic_trans = None  # 圆柱绕流不需要
        self.dirichlet_trans = Dirichlet()
        self.inlet_trans = DirichletInlet()
        self.neumann_trans = None   # 圆柱绕流不需要
        self.node_type_trans = NodeTypeInfo()
        self.mask_face_trans = MaskFace()
        
        # 2. 创建图结构变换链
        self.graph_trans = T.Compose([
            T.Delaunay(),           # Delaunay三角剖分
            self.mask_face_trans,   # 移除圆柱内部三角形
            T.FaceToEdge(),         # 面转边
            MyDistance(norm=True),  # 计算边长
            MyCartesian(norm=True), # 计算边方向
        ])
        
        # 3. 组合所有变换
        transform = []
        if self.dirichlet_trans is not None:
            transform.append(self.dirichlet_trans)
        if self.inlet_trans is not None:
            transform.append(self.inlet_trans)
        if self.periodic_trans is not None:
            transform.append(self.periodic_trans)
        if self.neumann_trans is not None:
            transform.append(self.neumann_trans)
        transform.append(self.node_type_trans)  # 节点类型标记
        transform.append(self.graph_trans)       # 图结构构建
        
        # 4. 传递给父类作为预变换
        super().__init__(
            root=root,
            transform=None,
            pre_transform=T.Compose(transform),  # 预处理时执行
            pre_filter=None
        )
        ...
```

---

### 4.2 完整执行流程

```python
# 第一步：原始数据准备
raw_data = {
    'pos': 节点坐标,
    'U': 速度场时间序列,
    'node_type': {'inlet': [...], 'cylinder': [...], 'outlet': [...]}
}

# 第二步：数据对象初始化
data = Data(pos=raw_data['pos'], y=raw_data['U'])

# 第三步：按顺序应用变换

# 1. Dirichlet() → 标记圆柱壁面节点
data.dirichlet_index = cylinder_index

# 2. DirichletInlet() → 标记入口节点
data.inlet_index = inlet_index

# 3. NodeTypeInfo() → 标记所有节点类型
data.node_type = [OBSTACLE, NORMAL, INLET, NORMAL, OUTLET, ...]

# 4. Delaunay() → 构建三角网格
data.face = [[0, 1, 2], [1, 3, 4], ...]  # 三角形顶点索引

# 5. MaskFace() → 移除圆柱内部三角形
data.face = [[0, 1, 2], [1, 3, 4], ...]  # 过滤后的三角形

# 6. FaceToEdge() → 转换为边连接
data.edge_index = [[0, 1, 0, 2, ...], [1, 0, 2, 0, ...]]  # 双向边

# 7. MyDistance() → 计算边长
data.edge_attr = [[dist_01], [dist_10], [dist_02], [dist_20], ...]

# 8. MyCartesian() → 计算边方向并拼接
data.edge_attr = [[dist_01, Δx_01, Δy_01], [dist_10, Δx_10, Δy_10], ...]
# 最终形状: (num_edges, 3)

# 第四步：保存处理后的数据
torch.save((data, slices), 'processed/data.pt')
```

---

## 📊 五、数据流示意图

```
原始HDF5数据
    ├── pos: [1598, 2] - 节点坐标
    ├── U: [2000, 1598, 2] - 速度场
    └── node_type: 边界信息
    ↓
【数据处理】
    ├── 时间窗口切片
    ├── 无量纲化
    └── 噪声增强
    ↓
【Transform管道】
    ├── Dirichlet() → 标记边界节点
    ├── DirichletInlet() → 标记入口节点
    ├── NodeTypeInfo() → 节点类型编码
    ├── Delaunay() → 三角网格构建
    ├── MaskFace() → 移除无效三角形
    ├── FaceToEdge() → 边连接生成
    ├── MyDistance() → 边特征(距离)
    └── MyCartesian() → 边特征(方向)
    ↓
图神经网络输入
    ├── edge_index: [2, E] - 边连接
    ├── edge_attr: [E, 3] - 边特征
    ├── pos: [1598, 2] - 节点坐标
    ├── y: [1598, T, 2] - 速度场序列
    ├── node_type: [1598] - 节点类型
    └── 边界条件索引
```

---

## 🎯 六、与论文的对应关系

| Transform 类 | 论文概念 | 物理意义 | 在模型中的作用 |
|-------------|---------|---------|---------------|
| `MyCartesian` | 边方向特征 $e_{ij}$ | 几何连接关系 | 消息传递的相对位置信息 |
| `MyDistance` | 边长度 $|e_{ij}|$ | 距离衰减权重 | 影响消息传播的强度 |
| `Dirichlet` | 固定值边界 | 无滑移条件 | 强制约束边界值 |
| `DirichletInlet` | 入口边界 | 固定入口速度 | 提供流动驱动力 |
| `Periodic` | 周期性边界 | 空间连续性 | 处理周期性流动 |
| `Neumann` | 零梯度边界 | 自由出流 | 避免边界反射 |
| `MaskFace` | 障碍物处理 | 物理合理性 | 移除无效内部区域 |
| `NodeTypeInfo` | 节点分类 | 边界类型标记 | 指导边界条件应用 |

---

## 💡 七、使用建议与最佳实践

### 7.1 Transform 设计原则

1. **单一职责**：每个Transform只做一件事
2. **可组合性**：支持通过T.Compose组合使用
3. **状态独立**：避免Transform之间的隐式依赖
4. **文档完整**：清晰说明输入输出和副作用

### 7.2 性能优化

1. **缓存机制**：使用InMemoryDataset避免重复计算
2. **并行处理**：多个独立Transform可并行化
3. **惰性计算**：延迟计算直到真正需要时

### 7.3 调试技巧

1. **逐步验证**：单独测试每个Transform
2. **可视化检查**：绘制中间结果
3. **边界测试**：测试极端情况

---

## 🔧 八、扩展与自定义

### 8.1 添加新Transform

```python
@functional_transform('my_custom')
class MyCustomTransform(BaseTransform):
    def __init__(self, param1=1.0, param2=True):
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, data):
        # 实现自定义逻辑
        new_feature = compute_feature(data)
        data.new_feature = new_feature
        return data
    
    def __repr__(self):
        return f'MyCustomTransform(param1={self.param1}, param2={self.param2})'
```

### 8.2 适配其他物理问题

对于非流体力学问题，可以：

1. **复用几何Transform**：MyDistance、MyCartesian
2. **修改边界Transform**：根据新的边界条件
3. **添加物理Transform**：计算问题相关的物理量
4. **调整组合顺序**：根据具体需求

---

## 📚 九、总结

### 核心概念

1. **Transform本质**：对图数据进行操作的可调用对象
2. **BaseTransform**：所有Transform的抽象基类
3. **functional_transform**：提供函数式调用的装饰器
4. **T.Compose**：组合多个Transform的管道

### 设计优势

1. **模块化**：每个Transform独立可测试
2. **可复用**：Transform可在不同场景复用
3. **可扩展**：易于添加新的Transform
4. **可组合**：通过Compose灵活组合

### 在PhyMPGN中的作用

1. **数据转换**：从CFD数据到图格式
2. **特征工程**：计算几何和物理特征
3. **边界处理**：编码各类边界条件
4. **预处理优化**：一次计算多次使用

这个Transform系统为PhyMPGN提供了灵活、高效的数据处理管道，是实现物理编码学习的重要基础设施。