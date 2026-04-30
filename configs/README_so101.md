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
| `dataset.repo_id` | `TODO_user/so101_pick_lift_cube` | 改成你的 HF 帳號 + 任務名 |
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

## 切換到 SO101 leader teleop(取代 gamepad)

如果想用 SO101 leader 手臂遙操(比 gamepad 順):

```diff
   "teleop": {
-    "type": "gamepad",
-    "use_gripper": true
+    "type": "so101_leader",
+    "port": "/dev/ttyACM1",
+    "use_degrees": true
   },
   ...
   "processor": {
-    "control_mode": "gamepad",
+    "control_mode": "leader",
```

注意:用 leader 時,episode success / fail / rerecord / 介入 等仍然要靠 **鍵盤** — `s` = success、`esc` = fail、`space` = 介入 / 還回控制。

## 跑指令

```bash
cd ~/lerobot
uv run python -m lerobot.rl.gym_manipulator --config_path configs/env_config_so101.json
```
