# LeRobot HIL-SERL 安裝筆記

針對 SO101 + HIL-SERL 教程([huggingface.tw/docs/lerobot/hilserl](https://huggingface.tw/docs/lerobot/hilserl)),整理出實際能跑通的安裝步驟。教程本身有些坑沒寫到,這邊一併記錄。

> 適用:Ubuntu 24.04 LTS / Python 3.12。

---

## 1. 系統套件

裝 Python 開發 headers 與 C++ 編譯工具鏈。`placo` 系列(逆運動學)在 PyPI **只有 sdist 沒有 wheel**,一定要本機編譯;`evdev` 也需要 Python headers。

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    pkg-config \
    libssl-dev \
    libudev-dev \
    python3-dev \
    python3.12-dev \
    libeigen3-dev \
    ca-certificates
```

> **重點**:`python3-dev` 一定要裝。少了它,`evdev` 編譯會炸 `fatal error: Python.h: No such file`。

## 2. 安裝 uv

LeRobot 官方推薦用 [uv](https://github.com/astral-sh/uv) 管理依賴(比 pip 快,且可管 Python toolchain)。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

uv 會裝到 `~/.local/bin/uv`。新 shell 會自動把它加進 PATH;若沒有,執行 `source ~/.local/bin/env`。

## 3. Clone repo

```bash
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

## 4. 釘住 Python 3.12(重要)

`pyproject.toml` 寫 `requires-python = ">=3.12"`,uv 預設會抓最新可用的 Python(目前有 3.13、3.14)。但好幾個關鍵套件目前在 PyPI **只有 cp312 / cp313 wheel,沒有 cp314**(例如 `hidapi`),會被迫從 source 編而失敗。

```bash
echo "3.12" > .python-version
```

## 5. 安裝依賴

官方教程寫 `pip install -e ".[hilserl]"`,但**這還不夠**:

- `gym_manipulator` 模組會 `import LeRobotDataset`,需要 `dataset` extra
- SO101 馬達需要 Feetech SDK,要 `feetech` extra

```bash
uv sync --locked --extra hilserl --extra dataset --extra feetech
```

各 extra 帶的東西:

| extra | 主要套件 | 用途 |
|---|---|---|
| `hilserl` | `placo`, `pinocchio`, `cmeel-*`, `gym-hil`, `transformers`, `grpcio` | RL 的核心:IK、actor-learner gRPC、reward classifier |
| `dataset` | `datasets`, `pyarrow`, `pandas`, `av`, `torchcodec` | LeRobotDataset 資料集 / 影片解碼 |
| `feetech` | `feetech-servo-sdk`, `pyserial` | SO100/SO101 馬達 driver |

首次 sync 會花較久(下載 ~1.4 GB:torch + CUDA libs;另需要本機編譯 `placo` 的 C++ 依賴),預期 5–10 分鐘。

## 6. 驗證

```bash
uv run python -c '
import lerobot
import torch
import placo
from lerobot.envs import HILSerlRobotEnvConfig
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig
print("lerobot:", lerobot.__version__)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("placo / SO101 / hilserl env / make_robot_env: ok")
'
```

預期輸出:

```
lerobot: 0.5.2
torch: 2.10.0+cu128 cuda: True
placo / SO101 / hilserl env / make_robot_env: ok
```

## 7. 跑 HIL-SERL

```bash
cd ~/lerobot
uv run python -m lerobot.rl.gym_manipulator --config_path <你的 env_config.json>
```

> ⚠️ 教程指令裡的 `src/lerobot/configs/env_config_so101.json` **這個檔案不存在於 repo**(教程預設你會自己建)。範本可參考 SO100 範例:[aractingi/lerobot-example-config-files/env_config_so100.json](https://huggingface.co/datasets/aractingi/lerobot-example-config-files/blob/main/env_config_so100.json),把 `so100_*` 改 `so101_*`,並填上 USB port、camera、URDF 路徑。SO101 leader 教程章節有給 `control_mode: "leader"` 的設定範例。

---

## 已知踩雷點(教程沒寫的)

| 症狀 | 根因 | 解法 |
|---|---|---|
| 編 `hidapi` 失敗(Windows 顯示「Microsoft Visual C++ 14.0 required」;Linux 也可能少 wheel) | uv 自動拉了 Python 3.14,PyPI 上 hidapi 沒 cp314 wheel,被迫源碼編 | `.python-version` 釘 3.12 |
| 編 `evdev` 失敗:`fatal error: Python.h: No such file` | 系統缺 Python dev headers | `sudo apt install python3-dev` |
| 編 `cmeel-console-bridge / cmeel-urdfdom / placo` 失敗 | 這些 C++ 套件沒 Windows wheel,要 MSVC + CMake | 改 Linux,或裝 [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| `ImportError: 'datasets' is required` | `gym_manipulator` 用到 `LeRobotDataset`,但 hilserl extra 沒含 datasets | 加 `--extra dataset` |
| 找不到 `env_config_so101.json` | 教程命令引用範本檔,但 repo 沒附 | 以 SO100 範例為底自行建立 |

## 已準備好(這台電腦,無需手臂)

- [x] uv 環境裝好,placo / SO101 configs / hilserl env / make_robot_env 全部 import OK
- [x] **SO101 URDF + 13 個 STL meshes** 下載至 `assets/so101/`(來自 [TheRobotStudio/SO-ARM100/Simulation/SO101](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101),16 MB)
- [x] placo `RobotKinematics` 已驗證可正確 load URDF,joint_names = `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`,FK at zero pose 給合理 EE 位置
- [x] **`configs/env_config_so101.json`** 寫好(新版嵌套 schema、gamepad teleop),draccus 解析通過 — 全 schema 對得上 dataclass(教程文件範例的 `type: gym_manipulator` 與 `task / fps` 在 env 層級已過期,本範本已修正)
- [x] **`configs/env_config_so101_leader.json`** + 自製 **`so101_leader_hil`** teleop class(`src/lerobot/teleoperators/so_leader_hil/`)— 補上 upstream 沒接通的 leader+keyboard HIL-SERL 流程(leader 帶 follower、鍵盤標 success/fail/intervention)。draccus 解析、factory 實例化、action_features 全部驗證通過(待硬體實測 leader 連線與 pynput 鍵盤)
- [x] **`configs/env_config_so101_reward.json`** — reward classifier 資料蒐集用(`terminate_on_success: false` + 獨立 `repo_id/root`)
- [x] **`configs/reward_classifier_train_config.json`** — `lerobot-train` 訓練 reward classifier 用(`reward_model.type=reward_classifier`, ResNet10, 2 cameras × 128×128, 5000 steps);draccus 解析通過
- [x] **`configs/train_config_hilserl_so101.json`** — actor-learner 訓練主 config(`policy.type=sac`, `num_discrete_actions=3` 給 gripper、connection on `127.0.0.1:50051`、合併整段 `env` HILSerlRobotEnvConfig);draccus 解析通過
- [x] **`configs/README_so101.md`** 註明每個欄位該填什麼、用什麼指令量,以及 gamepad ↔ leader+keyboard 的選擇對照

## 接到 SO101 才能做的事(在公司 PC)

按順序做:

1. **找 USB port** — `uv run lerobot-find-port`(分別插拔 follower / leader)→ 寫進 `configs/env_config_so101.json` 的 `env.robot.port`(以及 leader 的 `env.teleop.port` 如果用 leader 取代 gamepad)。
   - Linux 通常 `/dev/ttyACM0`、`/dev/ttyACM1`,記得 `sudo chmod 666 /dev/ttyACM*` 給存取權
2. **設定馬達 ID 與 baudrate**(全新馬達一次性):`uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACMx`,leader 同理
3. **找相機 index** — `ls /dev/video*` 或 `v4l2-ctl --list-devices`,測試後填入 `cameras.{front,wrist}.index_or_path`
4. **找 end-effector 工作範圍**(用 leader 帶動 follower 走過任務區域):
   ```bash
   uv run lerobot-find-joint-limits \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM0 \
     --robot.id=follower \
     --teleop.type=so101_leader \
     --teleop.port=/dev/ttyACM1 \
     --teleop.id=leader
   ```
   - 跑起來後 follower 會解力,**抓 leader 引導 follower 走遍要解任務需要走過的所有位置**(起點 / 目標 / 中間轉折點都要走到)
   - Ctrl+C 結束,會印出類似:
     ```
     Max ee position [0.2417 0.2012 0.1027]
     Min ee position [0.1663 -0.0823 0.0336]
     Max joint positions [...]
     Min joint positions [...]
     ```
   - 把 `Max ee position` / `Min ee position` 三個浮點數填回 `configs/env_config_so101.json` 的 `inverse_kinematics.end_effector_bounds.max` / `.min`
   - **小盒子比大盒子好**(訓練快很多);包足任務區域後留 1–2 cm 餘裕、`z` 下界 ≥ 桌面 + 1 cm 防撞桌
   - 範例對比:目前佔位符是 `±1.0 m` 立方體(完全沒約束),教程 SO100 範例是 `~8 × 28 × 7 cm` 小盒,差超過 100 倍,請務必縮到任務範圍

5. **找 reset 姿勢**(每集 episode 從哪個姿勢開始):

   ```bash
   uv run lerobot-teleoperate \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM0 \
     --robot.id=follower \
     --robot.use_degrees=true \
     --teleop.type=so101_leader \
     --teleop.port=/dev/ttyACM1 \
     --teleop.id=leader \
     --teleop.use_degrees=true \
     --display_data=true
   ```

   - 用 leader **把 follower 帶到「每集應該從哪開始」的姿勢**(例如 EE 懸在工作區正中央上方 ~10 cm,gripper 微開)
   - 終端機會即時印 6 個關節度數(NORM 欄位):
     ```
     NAME           | NORM
     shoulder_pan   |   0.00
     shoulder_lift  |   0.00
     elbow_flex     |   0.00
     wrist_flex     |  90.00
     wrist_roll     |   0.00
     gripper        |   5.00
     ```
   - 把這 6 個值**按上面順序**填回所有 4 個 config 的 `env.processor.reset.fixed_reset_joint_positions`(`env_config_so101.json` / `_leader.json` / `_reward.json` / `train_config_hilserl_so101.json`),Ctrl+C 退出
   - 條件:這個姿勢必須**落在 step 4 找出的 end_effector_bounds 之內**(否則 reset 完馬上被 IK 拉回 bounds 邊界,動作會抖)

6. **錄 demonstrations**:選一種 teleop 開跑

   **A. Gamepad mode**(預設,穩定):
   ```bash
   cd ~/lerobot
   uv run python -m lerobot.rl.gym_manipulator --config_path configs/env_config_so101.json
   ```
   按鈕對映:Y/△ = success(reward=1)、A/× = failure(reward=0)、X/□ = rerecord、RB(按住)= intervention、LT/RT-area = gripper open/close

   **B. Leader + Keyboard mode**(需要 leader 手臂):
   ```bash
   cd ~/lerobot
   uv run python -m lerobot.rl.gym_manipulator --config_path configs/env_config_so101_leader.json
   ```
   鍵盤對映:`s` = success、`esc` = failure、`r` = rerecord、`space` = toggle intervention。動作來自 leader(透過 FK 轉成 EE-delta)。注意 `end_effector_step_sizes` 在 teleop 區段跟 `processor.inverse_kinematics` 區段必須相同,否則人控 scale 會跟 policy 對不起來。詳見 `configs/README_so101.md`。

   通用流程(兩種 mode 共通):
   - 預期 ≥15 個 episode,每集 5–10 秒
   - 完成後接 step 7 框 ROI(再進到訓練)

7. **框 ROI + 縮圖**(把訓練輸入縮到只看任務區域):

   ```bash
   cd ~/lerobot
   uv run python -m lerobot.rl.crop_dataset_roi \
     --repo-id local/so101_pick_lift_cube \
     --root data/so101_pick_lift_cube \
     --task pick_and_lift
   ```

   會跳出 OpenCV 視窗,**對每個相機畫一個方框**(滑鼠拖曳)→ Enter 確認 → ESC 跳下一個。產出:
   - 新資料集:`data/so101_pick_lift_cube_cropped_resized/`(repo_id `local/so101_pick_lift_cube_cropped_resized`)
   - ROI 參數:`<新資料集>/meta/crop_params.json`(把這個 dict 內容貼到 `train_config_hilserl_so101.json` 的 `env.processor.image_preprocessing.crop_params_dict`,actor 在訓練時就會即時 crop 同範圍)

8. **(選)訓練 reward classifier** — 自動偵測 episode 成功,可省下人手按 success。沒做的話 actor 期間每集仍要按 RB/`s` 標 success,policy 還是學得起來。

   a. **錄 reward classifier 資料集**(成功+失敗各半,episode 多但短):
   ```bash
   cd ~/lerobot
   uv run python -m lerobot.rl.gym_manipulator --config_path configs/env_config_so101_reward.json
   ```
   差別 vs `env_config_so101.json`:`terminate_on_success: false`(成功後不結束,多收成功 frame)、`repo_id/root` 改 `so101_reward_data`。建議錄 30+ episodes,各 5–10 秒。

   b. **訓練 classifier**:
   ```bash
   uv run lerobot-train --config_path configs/reward_classifier_train_config.json
   ```
   產出:`outputs/train/so101_reward_classifier/checkpoints/last/pretrained_model/`

   c. **接到 actor-learner**:把 `train_config_hilserl_so101.json` 的 `env.processor.reward_classifier.pretrained_path` 從 `null` 改成上面那個 path。

9. **訓練 actor-learner**(SAC RL,需要兩個 terminal):

   開兩個 terminal,**先 learner、後 actor**(actor 透過 gRPC 連 learner)。兩邊用同一個 config:

   Terminal 1 — Learner:
   ```bash
   cd ~/lerobot
   uv run python -m lerobot.rl.learner --config_path configs/train_config_hilserl_so101.json
   ```

   Terminal 2 — Actor(等 learner 印出 `Listening on 127.0.0.1:50051` 再開):
   ```bash
   cd ~/lerobot
   uv run python -m lerobot.rl.actor --config_path configs/train_config_hilserl_so101.json
   ```

   訓練期間:
   - actor 跑 policy → 蒐集 transitions → gRPC 送 learner → learner 更新 → 推回 actor
   - 用 gamepad RB 按住做 intervention(rollout 開頭多干預幾次,中後期讓 policy 自己跑)
   - W&B:`train_config_hilserl_so101.json` 頂層的 `wandb.enable: true` + 設定 `WANDB_API_KEY`,可看 `intervention_rate`、`episode_reward` 隨 step 下降/上升的曲線
   - checkpoint 存在 `outputs/train/so101_hilserl/checkpoints/`,`save_freq=10000`
