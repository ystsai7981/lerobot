# `env_config_so101.json` 使用說明

供 `python -m lerobot.rl.gym_manipulator` 使用的 SO101 HIL-SERL 環境設定。範本以新版嵌套 schema(`env.processor.*`)寫成,參照 `docs/source/hilserl.mdx`。

## 接上 SO101 之前 → 必須填入真值的欄位

| 路徑 | 目前佔位符 | 取得方式 |
|---|---|---|
| `env.robot.port` | `/dev/ttyACM0` | `lerobot-find-port`(拔掉 follower → 看消失的那個) |
| `env.robot.cameras.front.index_or_path` | `0` | `v4l2-ctl --list-devices` 或 `ls /dev/video*` 試出來 |
| `env.robot.cameras.wrist.index_or_path` | `2` | 同上 |
| `env.processor.inverse_kinematics.end_effector_bounds` | `±1.0`(很鬆) | `lerobot-find-joint-limits` 量真實工作範圍後填回(min/max 各 3 個浮點:x,y,z,單位 m) |
| `env.processor.reset.fixed_reset_joint_positions` | `[0,0,0,90,0,5]` | 手臂上電後手動移到「初始姿勢」,讀關節度數填入 |
| `dataset.repo_id` | `local/so101_pick_lift_cube` | 純本地用任意 `<owner>/<name>`(必須有斜線);要 push 上 HF 才填 `<你的HF帳號>/<任務名>` |
| `dataset.root` | `data/so101_pick_lift_cube` | 資料存放位置;`data/` 已 gitignore,不會被 commit;設成 `null` 就改放 HF cache (`~/.cache/huggingface/lerobot/<repo_id>`) |
| `dataset.push_to_hub` | `false` | `true` 才會在錄完後上傳 HF Hub |
| `dataset.task` | `pick_and_lift` | 任務識別字串(自己取) |

## 範本選擇 (對應流程階段)

`mode` 跟 `dataset.push_to_hub` 是切換流程的主開關:

| 階段 | `mode` | `push_to_hub` | 其他要改的 |
|---|---|---|---|
| 錄 demonstrations | `"record"` | `true`(若要傳上 Hub) | `dataset.num_episodes_to_record`(建議 15–30) |
| 錄 reward classifier 資料 | `"record"` | `true` | `env.processor.reset.terminate_on_success: false`(讓 episode 收滿成功 frame) |
| 訓練(actor-learner) | `null` | `false` | `env.processor.reward_classifier.pretrained_path` 填訓練好的 reward 模型 |
| Replay | `"replay"` | — | `dataset.replay_episode` 設要重播的 episode index |

## URDF & IK

URDF 已下載至 `assets/so101/so101_new_calib.urdf`(來自 [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101))。

- `target_frame_name: "gripper_frame_link"` — placo 用的 EE frame 名,**不要改**(URDF 內定義)
- 6 關節順序:`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
- placo IK solver 已驗證可正確 load URDF 並做 FK(這台電腦上跑過)

## 切換 teleop:gamepad ↔ leader+keyboard

預設 `env_config_so101.json` 用 **gamepad**(穩定、現成可用)。如果你也有 SO101 leader 手臂,可改用 **`env_config_so101_leader.json`** — 用 leader 帶動 follower、用鍵盤標 success/failure。

| | gamepad (`env_config_so101.json`) | leader+keyboard (`env_config_so101_leader.json`) |
|---|---|---|
| Teleop type | `"gamepad"` | `"so101_leader_hil"` |
| 動作來源 | 左/右搖桿 + 觸發鍵 | 用手帶 leader 跑(透過 FK 轉成 EE-delta) |
| Episode 事件 | gamepad 按鈕(Y/A/X) | 鍵盤(`s` / `esc` / `r`) |
| Intervention 切換 | RB 按住 | 鍵盤 `space` 切換 (toggle) |
| Gripper | 觸發鍵 OPEN/CLOSE | leader gripper 角度變化(可調 threshold) |
| 教程支援程度 | 官方 | 我們的 fork 才有(upstream 沒接通)|

> ⚠️ 重要:upstream 教程說可以 `control_mode: "leader"` 配合 `so101_leader` teleop,但實際 code **沒實作 `get_teleop_events()`**,跑下去會 `TypeError`。我們這個 `so101_leader_hil` 是補上那塊空缺(leader 動作 + 鍵盤事件 + FK 把關節轉 EE-delta)。

### 用 leader+keyboard mode 的指令

```bash
cd ~/lerobot
uv run python -m lerobot.rl.gym_manipulator --config_path configs/env_config_so101_leader.json
```

### 鍵盤對映(只在 `so101_leader_hil` mode 下)

| 鍵 | 功能 |
|---|---|
| `s` | SUCCESS(reward=1,結束 episode) |
| `esc` | FAILURE(reward=0,結束 episode) |
| `r` | RERECORD(丟掉這集重來) |
| `space` | toggle INTERVENTION(切換 leader 是否覆蓋 policy) |

### Leader+keyboard config 額外參數(只在 `so101_leader_hil` 用得到)

| 欄位 | 預設 | 說明 |
|---|---|---|
| `urdf_path` | `assets/so101/so101_new_calib.urdf` | 給 placo 做 FK 用,leader 跟 follower 同 URDF |
| `target_frame_name` | `gripper_frame_link` | URDF 內 EE frame 名(不要改) |
| `end_effector_step_sizes` | `{x:.025, y:.025, z:.025}` | **必須跟 `processor.inverse_kinematics.end_effector_step_sizes` 一致**,否則人控 scale 會跟 policy 不同 |
| `gripper_open_threshold_deg` | `1.0` | leader gripper 一幀內度數變化 ≥ 此值 → 命令 OPEN |
| `gripper_close_threshold_deg` | `-1.0` | ≤ 此值 → CLOSE,介於兩者 → STAY |

## 跑指令

```bash
cd ~/lerobot
uv run python -m lerobot.rl.gym_manipulator --config_path configs/env_config_so101.json
```
