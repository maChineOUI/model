# TP TM — PDE 完整任务说明（结构化版本）

---

## 0. 问题定义

求解如下偏微分方程：

    u_t(x,t) = u_xx(x,t) - α u(x,t),     x ∈ (0,1), t > 0

边界条件（Dirichlet）：
    u(0,t) = 0
    u(1,t) = 0

初始条件：
    u(x,0) = f(x)

---

## 1. 解析解（Solution analytique）

---

### 1(a) 单一模态初值

初值：
    f₁(x) = sin(2πx)

任务：

- 使用变量分离法（séparation des variables）求解析解
- 写出完整形式 u(x,t)
- 验证：
    - 满足 PDE
    - 满足边界条件
    - 满足初始条件

---

### 1(b) 分段函数初值

初值：
    f₂(x) =
        2x           if 0 ≤ x ≤ 1/2
        2(1-x)       if 1/2 < x < 1

任务：

- 将 f₂ 展开为 Fourier 正弦级数：
      f₂(x) = Σ b_n sin(nπx)
- 计算系数 b_n
- 写出解析解：

      u(x,t) = Σ b_n exp(-((nπ)² + α)t) sin(nπx)

---

## 2. 数值解：显式格式（Schéma explicite）

---

### 2(a) 构造格式

要求：

- 时间：一阶（ordre 1 en temps）
- 空间：二阶（ordre 2 en espace）

任务：

- 使用：
    - 时间前向差分
    - 空间中心差分
- 写出完整差分格式：
    u_i^{n+1} = ...

---

### 2(b) 稳定性分析

任务：

- 使用 Von Neumann 方法
- 引入傅里叶模态：
    e^{ikx}
- 推导放大因子 G
- 得出稳定性条件（CFL条件）：
    r = Δt / Δx² 满足某约束

---

### 2(c) 矩阵形式

任务：

- 定义向量：
    U^n = (u₁^n, ..., u_{N-1}^n)^T
- 写成：
    U^{n+1} = A U^n
- 给出矩阵 A 的结构：
    - 三对角矩阵
    - 明确对角线、上下对角元素

---

### 2(d) 编程实现

任务：

- 使用 Scilab / Matlab / Python
- 实现显式时间推进
- 输入：
    Nx, Nt, α, T, 初始条件
- 输出：
    u(x,t)

---

## 3. 数值解：隐式格式（Schéma implicite）

---

### 3(a) 构造格式

要求：

- 时间一阶
- 空间二阶

任务：

- 使用后向欧拉（Backward Euler）
- 写出差分格式：
    使用 n+1 层空间导数

---

### 3(b) 稳定性分析

任务：

- 使用 Von Neumann 方法
- 推导放大因子
- 证明：
    无条件稳定（inconditionnellement stable）

---

### 3(c) 矩阵形式

任务：

- 写成线性系统：
    B U^{n+1} = U^n
- 明确矩阵 B：
    - 三对角
    - 写出系数

---

### 3(d) 编程实现

任务：

- 每一步求解线性系统：
    U^{n+1} = B^{-1} U^n
- 推荐：
    - Thomas 算法（tridiagonal solver）
    - 或 numpy.linalg.solve

---

## 4. 数值解：Crank–Nicholson 格式

---

### 4(a) 构造格式

任务：

- 将显式与隐式格式取平均
- 写出完整差分格式

---

### 4(b) 阶数分析

任务：

- 说明时间与空间阶数：
    O(Δt² + Δx²)

---

### 4(c) 稳定性分析

任务：

- 使用 Von Neumann 方法
- 说明稳定性：
    无条件稳定（但可能振荡）

---

### 4(d) 矩阵形式

任务：

- 写成：
    A U^{n+1} = B U^n
- 明确：
    - A 矩阵
    - B 矩阵
    - 三对角结构

---

### 4(e) 编程实现

任务：

- 每一步：
    解线性系统
- 与隐式类似，但右端不同

---

## 5. 结果展示与分析

---

### 5(a) 数值结果展示

任务：

对以下组合分别计算并绘图：

- 三种格式：
    - 显式
    - 隐式
    - Crank–Nicholson

- 两种初始条件：
    - f₁
    - f₂

---

### 5(b) 对比分析

必须分析：

#### 1. 精度（précision）
- 与解析解对比（尤其 f₁）
- 误差大小

#### 2. 稳定性（stabilité）
- 显式格式是否爆炸
- 隐式/CN 是否稳定

#### 3. 数值耗散（diffusion numérique）
- 是否过度平滑

#### 4. 振荡现象
- CN 在大 Δt 下可能出现

#### 5. 计算成本
- 显式：快
- 隐式/CN：需要解线性系统

---

## 6. 推荐输出形式

---

### 图像：

- u(x,t) 曲线（多个时间）
- 数值 vs 解析（f₁）
- 三种方法对比

---

### 数值指标：

- L2 error
- max error

---

### 报告中应包含：

- 方法推导
- 稳定性结论
- 数值实验结果
- 讨论与结论

---

## 7. 关键词（法语）

- équation aux dérivées partielles (EDP)
- condition initiale
- condition aux bords
- schéma explicite
- schéma implicite
- schéma de Crank-Nicholson
- stabilité
- convergence
- forme matricielle
- implémentation
- erreur numérique