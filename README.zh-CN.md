# 双臂遥操作

[English](README.md)

这是一个面向双臂 JAKA K1 的 PICO 遥操作项目。当前运行链路保持精简：
XR 人体/手柄输入 -> SWGR 目标重定向 -> AEAC 肘部偏好 ->
堆叠 J-PARSE 速度 IK -> Viser 可视化。本项目没有物理仿真后端，也没有真实硬件输出链路。

## 概览

```
PICO 头显 + 手柄 + Swift 追踪器
        |  xrobotoolkit_sdk
        v
     XrClient
        |
        v
  每只手臂的 SWGR 目标位姿 + grip 使能
        |
        v
  One Euro / Slerp 目标位姿滤波
        |
        v
  ViserJparseController -> J-PARSE IK -> Viser
```

`BaseTeleopController` 负责 XR 读取、人体追踪检查、SWGR 映射和 grip 使能锁存。
`ViserJparseController` 负责目标滤波、AEAC/J-PARSE IK、输出插值和可视化。

## 安装

已在 Ubuntu 22.04 和 24.04 测试。建议使用 Python 3.10 或 3.11。

```bash
conda deactivate
conda create -n bimanual-teleop python=3.10 -y
conda activate bimanual-teleop
bash setup_conda.sh --install
```

安装脚本会在 `dependencies/` 下构建 XRoboToolkit Python binding，并以 editable
模式安装本项目。`xrobotoolkit_sdk` 只用于从 PICO/XRoboToolkit 获取输入数据；
IK 的运动学模型由外部 git 依赖提供。

`dependencies/` 是外部 SDK 的本地安装目录，已在 `.gitignore` 中忽略，不属于本项目源码。
如果只准备提交代码，可以删除它；重新运行 `bash setup_conda.sh --install` 会重新生成。

验证安装：

```bash
python -m pip check
python -c "import viser, yourdfpy, xrobotoolkit_sdk, bimanual_teleop; print('ok')"
python -m unittest discover tests
```

## 使用

先启动 XRoboToolkit PC Service，连接头显，并开启全身追踪，然后运行：

```bash
conda activate bimanual-teleop
python scripts/simulation/teleop_jaka_k1.py
```

两只手臂都使用 SWGR：操作者肩到腕的向量会按机器人/人体臂展比例
(`0.768 / 0.58`) 缩放，并锚定到机器人肩部。手柄姿态用于设置工具姿态。
按住左/右 grip 使能对应手臂；松开 grip 后保持最后一个目标。

常用参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--frequency` | `60.0` | 控制循环频率，单位 Hz。 |
| `--engage-ramp-time` | `0.08` | grip 按下后目标混合进入的时间。 |
| `--gamma` | `0.1` | J-PARSE 奇异性阈值。 |
| `--max-joint-velocity` | `6.0` | 关节速度限幅，单位 rad/s。 |
| `--position-filter-min-cutoff` | `12.0` | One Euro 位置滤波基础截止频率。 |
| `--position-filter-beta` | `20.0` | 目标快速运动时提高滤波截止频率。 |
| `--orientation-filter-alpha` | `0.98` | Slerp 姿态低通滤波系数。 |
| `--elbow-gain` | `0.0` | AEAC 肘部方向零空间增益；设为 `0.5` 可开启。 |
| `--elbow-deadband` | `0.04` | 人体肘部偏移低于该值时忽略肘部方向。 |
| `--elbow-weight-max` | `1.0` | AEAC 自适应权重上限。 |
| `--elbow-filter-alpha` | `0.35` | AEAC 肘部方向/权重低通系数。 |
| `--grip-on-threshold` | `0.9` | grip 高于该值时激活。 |
| `--grip-off-threshold` | `0.75` | grip 低于该值时释放，避免模拟量抖动反复重置。 |
| `--enable-output-interpolation` | `False` | 在 IK 更新之间插值显示关节输出。 |
| `--output-frequency` | `200.0` | 开启插值时的输出更新频率。 |
| `--visualization-frequency` | `30.0` | Viser 刷新频率；控制循环仍按 `--frequency` 运行。 |

排查延迟时，先保持 `--elbow-gain 0`。如果 Viser 界面卡顿，可以先试
`--visualization-frequency 20`。等末端主任务足够跟手后，再用 `--elbow-gain 0.5`
打开 AEAC。

## 目录结构

```
scripts/simulation/teleop_jaka_k1.py   运行入口
bimanual_teleop/core/                  后端无关的遥操作控制器
bimanual_teleop/io/                    XR SDK 封装
bimanual_teleop/retargeting/           SWGR + AEAC 几何
bimanual_teleop/ik/                    J-PARSE 速度 IK
bimanual_teleop/runtime/               Viser 运行控制器
bimanual_teleop/robots/                JAKA K1 配置和资源路径
assets/robot_model/                    JAKA K1 URDF 和 meshes
tests/                                 离线几何/滤波/模型测试
```

## License

MIT - 见 [LICENSE](LICENSE)。
