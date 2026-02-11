# 3D TTI RTM CUDA Demo (Internal Validation)

这是一个用于 **GPU 内部性能验证** 的闭环 3D TTI RTM demo，不是产品级地震成像软件。

## 功能概览

- 3D 二阶时间推进波动方程。
- 8 阶空间差分（`dx, dy, dz` 分别参与计算）。
- TTI 近似算子包含旋转方向基和交叉导数项（非 `scalar laplace * factor`）。
- 震源：Ricker 子波（`f0` 可调）。
- 吸收边界：海绵阻尼层（sponge）。
- 正演记录炮集（`NT x NX x NY`，接收面位于 `z=sz`）。
- 反传注入记录并执行零延迟互相关成像：
  \[
  I(x,y,z)=\sum_t U_f(x,y,z,t)\cdot U_b(x,y,z,t)
  \]
- 显式 checkpoint + 重构机制：不依赖“凭空存在”的 forward wavefield。

## 工程结构

- `main.cu`：流程控制（正演/反演/成像/性能统计/输出）
- `kernel.cuh`：CUDA kernels
  - `forward_step_kernel`
  - `absorb_boundary_kernel`
  - `inject_source_kernel`
  - `record_surface_kernel`
  - `backward_inject_kernel`
  - `imaging_kernel`
- `config.hpp`：参数配置
- `model.hpp`：模型构建与 Ricker 函数
- `tools/plot_image.py`：结果可视化

## 编译

```bash
nvcc -O3 main.cu -o rtm
```

## 运行

默认参数运行：

```bash
./rtm
```

指定参数示例：

```bash
./rtm --nx 96 --ny 96 --nz 96 --nt 300 --dt 0.001 --f0 12 --checkpoint 20
```

运行后输出：

- `image.bin`：最终 3D RTM 成像体（float32, shape=`nz, ny, nx`）

## 可视化

```bash
python3 tools/plot_image.py --nx 96 --ny 96 --nz 96 --z 48 --input image.bin --output image_slice.png
```

## 参数调节建议

- 网格：`NX, NY, NZ`（对应 `--nx --ny --nz`）
- 时间步：`NT`（`--nt`）
- 子波主频：`f0`（`--f0`）
- checkpoint 间隔：`--checkpoint`
  - 越小：存储更多、重构更少
  - 越大：存储更省、重构更重

## 稳定性与限制

- 程序会打印近似 CFL 指标用于快速检查稳定性。
- TTI 算子是面向 demo 的简化实现，用于验证内核吞吐和闭环流程，不用于生产级成像精度。
- 该版本侧重“可读 + 可扩展 + 可跑通”，可进一步替换为更严格的伪声学 TTI 双变量方程、CPML、以及更高效的 checkpoint/revolve 策略。

