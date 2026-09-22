如果你说的 **Isaac 是 NVIDIA Isaac Lab**，最重要的理解是：

> **一个 Isaac Lab 项目不是“一个模型”，也不只是“一个插件”，而是一整套机器人学习实验工程。**
> 里面通常同时包含 **Isaac/Omniverse Extension、机器人与场景配置、RL Task 定义、Agent/算法配置、训练/测试入口脚本**。

官方现在把一个模板项目理解成四层：**Project → Extension → Module → Task**。([Isaac Sim][1])

一个典型项目，可以先这样看：

```text
my_robot_project/                 # Project：整个 Git 仓库
│
├── README.md
├── pyproject.toml
│
├── source/                       # “定义东西”的地方
│   └── my_robot_project/         # Extension
│       ├── config/
│       │   └── extension.toml    # Extension 元数据
│       ├── setup.py
│       │
│       └── my_robot_project/     # Python Module
│           ├── __init__.py
│           │
│           ├── assets/           # 机器人/物体定义（可选）
│           │
│           └── tasks/
│               ├── manager_based/
│               │   └── my_task/
│               │       ├── __init__.py
│               │       ├── env_cfg.py
│               │       ├── mdp/
│               │       │   ├── rewards.py
│               │       │   ├── observations.py
│               │       │   ├── events.py
│               │       │   └── terminations.py
│               │       └── agents/
│               │           ├── rsl_rl_ppo_cfg.py
│               │           └── skrl_ppo_cfg.yaml
│               │
│               └── direct/
│                   └── my_task/
│                       ├── __init__.py
│                       ├── my_task_env.py
│                       └── agents/
│
├── scripts/                      # “运行东西”的地方
│   ├── reinforcement_learning/
│   │   └── rsl_rl/
│   │       ├── train.py
│   │       └── play.py
│   ├── demos/
│   └── tools/
│
├── logs/                         # 通常运行后生成
│   └── ...
│
└── data/ / assets/               # USD、mesh、数据等（项目自行组织）
```

官方模板本身也是这个思路：根目录是 Project，`source` 放 Python package / Extension，而 `scripts` 里放诸如 `train.py`、`play.py` 的运行入口。([Isaac Sim][1])

### 1. `source/...` 到底是不是“插件”？

**是，而且又不完全只是传统意义上的插件。**

Isaac Lab 建立在 Isaac Sim / Omniverse 的 Extension 系统之上。官方定义很明确：一个目录里如果有

```text
config/
└── extension.toml
```

它就可以作为一个 **Omniverse Extension**；Isaac Lab 自己本身也是由多个 Extension 组成的。([Isaac Sim][2])

比如官方 Isaac Lab：

```text
source/
├── isaaclab
├── isaaclab_assets
├── isaaclab_tasks
├── isaaclab_mimic
└── isaaclab_rl
```

这里每一个都可以理解成一个大的 Extension / Python package：

* `isaaclab`：核心仿真接口、机器人、传感器、执行器等
* `isaaclab_assets`：预配置机器人/资产
* `isaaclab_tasks`：预配置任务环境
* `isaaclab_mimic`：模仿学习数据生成相关
* `isaaclab_rl`：和各种 RL 框架对接的 wrapper。([Isaac Sim][3])

所以看到：

```text
source/foo/
    config/extension.toml
    setup.py
    foo/
```

你可以判断：

> **foo 是一个 Isaac/Omniverse Extension，同时也是一个 Python package。**

这和 VS Code 插件、浏览器插件的“插件”概念有点像——核心思想都是 **模块化、可加载、声明依赖**——但这里是 **Omniverse Extension**。

---

### 2. 那 `tasks/` 是什么？

这才是你做强化学习时最核心的地方。

可以把一个 Task 理解成：

> **“我要机器人解决什么问题”的完整定义。**

例如：

```text
tasks/
└── manager_based/
    └── pick_cube/
```

代表：

> 用某个机器人完成 Pick Cube 这个任务。

里面一般会定义：

```text
pick_cube/
├── __init__.py
├── pick_cube_env_cfg.py
├── mdp/
│   ├── observations.py
│   ├── actions.py
│   ├── rewards.py
│   ├── terminations.py
│   ├── events.py
│   └── commands.py
└── agents/
    └── rsl_rl_ppo_cfg.py
```

这里其实描述的是一个 MDP：

[
(\mathcal S,\mathcal A,P,R,\gamma)
]

也就是：

```text
机器人看到什么？           observation
       ↓
机器人可以做什么？         action
       ↓
什么叫做得好？             reward
       ↓
什么时候一局结束？         termination
       ↓
每局如何随机化？           event / reset
       ↓
机器人要追踪什么目标？     command
```

Isaac Lab 的 Manager-Based workflow 正是把这些东西拆成不同 Manager；官方示例里一个 RL task 会定义 scene、observations、actions、events、rewards、terminations、commands、curriculum 等。([Isaac Sim][4])

所以：

```text
tasks/xxx/
```

**不是神经网络模型定义。**

它更准确地说是：

> **RL Environment / MDP / Task Definition**

---

### 3. `env_cfg.py` 是不是“模型”？

也不是。

例如：

```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    robot = ROBOT_CFG
    ground = GroundPlaneCfg(...)
```

再比如：

```python
@configclass
class RewardsCfg:
    tracking = RewTerm(...)
    action_penalty = RewTerm(...)
```

以及：

```python
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene = MySceneCfg()
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
```

这是：

> **环境配置 / 实验定义。**

官方也特别推荐 Manager-Based workflow 通过 `ManagerBasedRLEnvCfg` 配置 Task，而不是自己改环境基类，这样 task specification 和 environment implementation 可以分离。([Isaac Sim][4])

可以把它理解成：

```text
env_cfg.py
    ↓
“我要搭一个什么世界”
“世界里有什么机器人”
“机器人看什么”
“机器人控制什么”
“奖励怎么算”
“一局多久”
“仿真 timestep 是多少”
“同时跑多少环境”
```

所以它不是 ML 中：

```python
class NeuralNetwork(nn.Module):
```

意义上的 model。

---

### 4. 那机器人模型放哪？

这是另一个容易混淆的“model”。

机器人模型一般由两部分组成：

```text
物理/视觉资产
    +
Isaac Lab 配置
```

例如：

```text
robot.usd
```

负责实际的：

```text
link
joint
mesh
collision
mass
inertia
material
...
```

然后 Python 里可能有：

```python
MY_ROBOT_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path="robot.usd"
    ),
    actuators={...},
    init_state=...
)
```

这个才可以叫：

> **Robot Model / Robot Asset Definition**

官方的 `isaaclab_assets` Extension 就负责预配置这些 assets，包括机器人等。([Isaac Sim][3])

因此需要区分三个完全不同的“模型”：

```text
robot.usd
       → 物理机器人模型

ROBOT_CFG / ArticulationCfg
       → Isaac Lab 对机器人的配置

policy neural network
       → RL 学出来的神经网络模型
```

它们不是一回事。

---

### 5. `agents/` 又是什么？

这里非常重要。

```text
agents/
├── rsl_rl_ppo_cfg.py
├── skrl_ppo_cfg.yaml
└── rl_games_ppo_cfg.yaml
```

通常不是 Task 本身，而是：

> **“我要用什么 RL 算法、什么网络、什么超参数训练这个 Task”。**

比如里面可能有：

```text
PPO
learning_rate = 3e-4
gamma = 0.99
num_steps_per_env = 24

actor:
    hidden_dims = [512, 256, 128]

critic:
    hidden_dims = [512, 256, 128]
```

所以可以理解成：

```text
env_cfg
↓
我要解决什么问题

agent_cfg
↓
我要让什么学习算法来解决

train.py
↓
真正启动学习
```

Gym 注册时，Isaac Lab 会把一个 Task ID 同环境配置和不同 RL backend 的 agent config 关联起来，例如 `env_cfg_entry_point`、`rsl_rl_cfg_entry_point`、`skrl_cfg_entry_point` 等。([Isaac Sim][5])

---

### 6. 那 `train.py` 是什么？

这个反而非常简单：

> **训练程序的入口 / launcher。**

比如：

```bash
python scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Ant-v0
```

官方 Isaac Lab 的 RL backend 就通过各自的 `train.py` 和 `play.py` 启动；`train.py` 根据 Task Name 找到注册好的环境和配置，然后创建环境、创建 agent，开始训练。([Isaac Sim][6])

所以通常**不建议把所有环境逻辑塞进 `train.py`**。

理想状态是：

```text
train.py
   │
   ├── 读取 task name
   │
   ├── 加载 env cfg
   │
   ├── 加载 agent cfg
   │
   ├── gym.make(...)
   │
   ├── 创建 PPO runner
   │
   └── learn()
```

也就是说：

> `train.py` 是发动汽车的钥匙，不是汽车本身。

---

## 最关键的一张关系图

你可以把整个 Isaac Lab 项目直接理解成：

```text
                    Isaac Lab Project
                           │
          ┌────────────────┴────────────────┐
          │                                 │
       source/                           scripts/
   “定义什么东西”                      “执行什么东西”
          │                                 │
          │                          ┌──────┴──────┐
          │                        train.py      play.py
          │                           │             │
          ▼                           │             │
      Extension                       │             │
          │                           │             │
      Python Module                   │             │
          │                           │             │
          ▼                           │             │
        Task ◀────────────────────────┘             │
          │                                         │
    ┌─────┼──────────┐                              │
    │     │          │                              │
 Scene    MDP      Agents                           │
    │     │          │                              │
    │     │          └── PPO / network / hyperparams
    │     │
    │     ├── observations
    │     ├── actions
    │     ├── rewards
    │     ├── terminations
    │     ├── commands
    │     └── events
    │
    ├── Robot
    ├── Objects
    ├── Terrain
    ├── Sensors
    └── Lights

                         train
                           │
                           ▼
                    checkpoint.pt
                           │
                           ▼
                         play.py
```

---

## 因此看到一个文件夹，可以这样快速判断

| 看到的东西                   | 它代表什么                                           |
| ----------------------- | ----------------------------------------------- |
| `config/extension.toml` | **Omniverse / Isaac Extension**                 |
| `source/foo/foo/...`    | Python module / Isaac 功能代码                      |
| `assets/`               | Robot、object、USD 等资产定义                          |
| `tasks/`                | **RL Task / Environment 定义**                    |
| `env_cfg.py`            | Environment / Scene / Task 配置                   |
| `mdp/rewards.py`        | Reward function                                 |
| `mdp/observations.py`   | Observation                                     |
| `mdp/actions.py`        | Action                                          |
| `mdp/terminations.py`   | Episode termination                             |
| `agents/`               | RL agent / algorithm 配置                         |
| `rsl_rl_ppo_cfg.py`     | PPO + policy network + training hyperparameters |
| `train.py`              | **训练入口**                                        |
| `play.py`               | checkpoint 推理 / evaluation                      |
| `.usd/.usda`            | 仿真世界/机器人/物体资产                                   |
| `logs/`                 | TensorBoard、checkpoint、训练结果                     |

所以回答你最开始那个问题：

**Isaac Lab 一个“项目”通常三种东西都有。**

```text
Project
│
├── Extension / Python package
│
├── Robot / Scene definition
│
├── Task / Environment definition
│
├── RL agent configuration
│
├── Training scripts
│
└── Checkpoints / Logs / Assets
```

其中最容易混的关系可以记成一句话：

> **Extension 是“代码怎么被 Isaac 加载”；Task 是“机器人要学什么”；Robot/Asset 是“仿真里有什么”；Agent config 是“用什么算法学”；train.py 是“开始学”。**

而 Isaac Lab 官方仓库本身也是这种思想：`source` 是组成 Isaac Lab 的 Extensions，`scripts` 是 standalone applications，包括环境运行、工具、教程以及强化/模仿学习工作流。([Isaac Sim][3])

如果你现在正在看的其实是某一个具体的 Isaac 项目目录，把它的 `tree -L 4` 结果贴给我，我可以**逐个目录/文件给你标注：“插件 / Robot 定义 / Task / Reward / PPO 配置 / 训练入口 / 输出模型”**，这样会比抽象讲目录结构更直观。

[1]: https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/project_structure.html "Project Structure — Isaac Lab Documentation"
[2]: https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/development.html "Extension Development — Isaac Lab Documentation"
[3]: https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/repo_structure.html "Repository organization — Isaac Lab Documentation"
[4]: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html "Creating a Manager-Based RL Environment — Isaac Lab Documentation"
[5]: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html?utm_source=chatgpt.com "Registering an Environment — Isaac Lab Documentation"
[6]: https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html "Quickstart Guide — Isaac Lab Documentation"
