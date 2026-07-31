# GPU 迁移运行指令

## 输出目录约定

后续新实验按类型分目录，避免所有结果堆在 `outputs/` 根目录：

```text
outputs/
  heuristic/
  priority_imitation/
  priority_rl/

checkpoints/
  priority_imitation/
  priority_rl/
```

旧版已经生成在 `outputs/` 根目录的文件可以保留作为历史证据。新命令统一写入
分层目录。

## 现在还需要执行的命令

如果你已经在 RTX 4060 笔记本上执行过前面的 clone、数据复制、MSE 训练和
`mse_pairwise` 训练，现在只需要从这里继续。下面命令会保留旧文件，并尽量把旧路径
结果迁移到新的分层目录。

```bash
# 1. 进入你已经拉取好的仓库
cd zju_zjui_vrp_assessment

# 2. 同步最新代码
git pull origin main

# 3. 创建新的分层输出目录
mkdir -p outputs/heuristic outputs/priority_imitation outputs/priority_rl checkpoints/priority_imitation checkpoints/priority_rl

# 4. 如果之前的 supervised imitation 结果还在旧根目录，迁移到新目录
# 如果目标文件已经存在，mv -n 不会覆盖
if [ -f checkpoints/priority_mse_rank.pt ]; then mv -n checkpoints/priority_mse_rank.pt checkpoints/priority_imitation/priority_mse_rank.pt; fi
if [ -f checkpoints/priority_mse_pairwise_rank.pt ]; then mv -n checkpoints/priority_mse_pairwise_rank.pt checkpoints/priority_imitation/priority_mse_pairwise_rank.pt; fi
if [ -f outputs/priority_mse_rank_summary.json ]; then mv -n outputs/priority_mse_rank_summary.json outputs/priority_imitation/priority_mse_rank_summary.json; fi
if [ -f outputs/priority_mse_pairwise_rank_summary.json ]; then mv -n outputs/priority_mse_pairwise_rank_summary.json outputs/priority_imitation/priority_mse_pairwise_rank_summary.json; fi
if [ -f outputs/predictions_priority_model.json ]; then mv -n outputs/predictions_priority_model.json outputs/priority_imitation/predictions_priority_mse.json; fi
if [ -f outputs/predictions_priority_mse_pairwise.json ]; then mv -n outputs/predictions_priority_mse_pairwise.json outputs/priority_imitation/predictions_priority_mse_pairwise.json; fi

# 5. 检查当前 RL 训练需要的文件是否存在
python3 -c "from pathlib import Path; files=['VRP_project/VRPData/train_data.pkl','VRP_project/VRPData/validation_data.pkl','VRP_project/VRPData/check_data_to_students.pkl','checkpoints/priority_imitation/priority_mse_pairwise_rank.pt']; print({name: Path(name).exists() for name in files})"

# 6. 检查 CUDA 是否可用
python3 -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"

# 7. 跑完整测试，确认最新代码在 4060 环境中正常
python3 -m pytest tests -v

# 8. 强化学习微调 smoke：先确认 RL 训练链路能跑通
python3 scripts/train_priority_rl.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --init-checkpoint checkpoints/priority_imitation/priority_mse_pairwise_rank.pt --checkpoint-output checkpoints/priority_rl/priority_rl_smoke.pt --summary-output outputs/priority_rl/smoke_summary.json --train-limit 16 --eval-limit 8 --epochs 1 --batch-size 8 --samples-per-instance 2 --temperature 1.0 --learning-rate 0.00001 --device cuda --no-postprocess-reward --no-postprocess-eval

# 9. 强化学习正式微调：从 pairwise imitation checkpoint 继续训练
python3 scripts/train_priority_rl.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --init-checkpoint checkpoints/priority_imitation/priority_mse_pairwise_rank.pt --checkpoint-output checkpoints/priority_rl/priority_rl_finetune.pt --summary-output outputs/priority_rl/rl_finetune_summary.json --epochs 20 --batch-size 32 --samples-per-instance 2 --temperature 1.0 --learning-rate 0.00001 --weight-decay 0.0001 --eval-limit 100 --device cuda --postprocess-reward --postprocess-eval

# 10. 在完整 validation 上评估 RL 微调模型
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/validation_data.pkl --checkpoint checkpoints/priority_rl/priority_rl_finetune.pt --limit 1000 --device cuda --postprocess

# 11. 对官方 check 数据生成 RL 预测结果
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/check_data_to_students.pkl --checkpoint checkpoints/priority_rl/priority_rl_finetune.pt --output outputs/priority_rl/predictions_priority_rl.json --device cuda --postprocess
```

完成后，把这三个文件复制回当前电脑同样的相对路径：

```text
checkpoints/priority_rl/priority_rl_finetune.pt
outputs/priority_rl/rl_finetune_summary.json
outputs/priority_rl/predictions_priority_rl.json
```

## 完整流程（从零开始）

```bash
# 以下命令用于在 RTX 4060 笔记本上从拉取仓库开始，运行到得到训练、评估和预测结果
# 本文只给项目运行命令，不包含 PyTorch 安装步骤

# 1. 拉取仓库
git clone https://github.com/HinataAsahi/zju_zjui_vrp_assessment.git

# 2. 进入项目目录
cd zju_zjui_vrp_assessment

# 3. 如果仓库已经存在，用这条命令同步最新代码
git pull origin main

# 4. 将官方数据目录放到仓库根目录
# 注意：下面命令里的 /实际/父目录 是占位路径，必须替换成你电脑上真实存在的目录
# 情况 A：如果你的数据目录外层叫 VRP_project，并且它里面有 VRPData
ls /实际/父目录/VRP_project/VRPData
cp -r /实际/父目录/VRP_project ./VRP_project

# 情况 B：如果你的数据在 Windows 下载目录中，并且你正在 WSL/Linux 里运行
# 把 YOUR_WINDOWS_USERNAME 替换成你的 Windows 用户名
ls /mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/VRP_project/VRPData
cp -r /mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/VRP_project ./VRP_project

# 情况 C：如果你已经手动把 VRP_project 放进当前仓库根目录，则跳过复制，只检查即可
ls VRP_project/VRPData

# 5. 检查三个数据文件是否在预期位置
python3 -c "from pathlib import Path; files=['train_data.pkl','validation_data.pkl','check_data_to_students.pkl']; base=Path('VRP_project/VRPData'); print({name:(base/name).exists() for name in files})"

# 6. 检查当前 Python 能否使用 PyTorch 与 CUDA；这里只检查，不安装
python3 -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"

# 7. 跑完整测试，确认迁移后的代码状态正常
python3 -m pytest tests -v

# 8. 先生成当前稳定 heuristic 的官方预测结果，作为安全提交版本
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions.json --device cuda

# 9. 用很小规模 smoke 训练检查训练脚本、数据读取、checkpoint 写入是否正常
# 训练时会默认显示实时 batch 进度条；如果想降低刷新频率，可额外加 --log-every 20
python3 scripts/train_priority_model.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --checkpoint-output checkpoints/priority_imitation/priority_smoke.pt --summary-output outputs/priority_imitation/priority_smoke_summary.json --train-limit 16 --eval-limit 8 --epochs 1 --batch-size 8 --hidden-dim 64 --num-heads 4 --num-layers 1 --dropout 0.1 --device cuda --no-postprocess-eval

# 10. 正式训练第一版客户优先级模型；输出 best checkpoint 和训练摘要
python3 scripts/train_priority_model.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --checkpoint-output checkpoints/priority_imitation/priority_mse_rank.pt --summary-output outputs/priority_imitation/priority_mse_rank_summary.json --epochs 50 --batch-size 64 --hidden-dim 128 --num-heads 4 --num-layers 2 --dropout 0.1 --learning-rate 0.001 --weight-decay 0.0001 --eval-limit 100 --device cuda --postprocess-eval

# 11. 在完整 validation 上评估训练出的优先级模型
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/validation_data.pkl --checkpoint checkpoints/priority_imitation/priority_mse_rank.pt --limit 1000 --device cuda --postprocess

# 12. 对官方 check 数据生成优先级模型预测结果，便于和 heuristic 输出对比
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/check_data_to_students.pkl --checkpoint checkpoints/priority_imitation/priority_mse_rank.pt --output outputs/priority_imitation/predictions_priority_mse.json --device cuda --postprocess

# 13. 正式训练第二版客户优先级模型：MSE + pairwise ranking loss
python3 scripts/train_priority_model.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --checkpoint-output checkpoints/priority_imitation/priority_mse_pairwise_rank.pt --summary-output outputs/priority_imitation/priority_mse_pairwise_rank_summary.json --epochs 50 --batch-size 64 --hidden-dim 128 --num-heads 4 --num-layers 2 --dropout 0.1 --learning-rate 0.001 --weight-decay 0.0001 --eval-limit 100 --device cuda --postprocess-eval --loss mse_pairwise --pairwise-weight 0.5 --pairwise-margin 0.1

# 14. 在完整 validation 上评估第二版优先级模型
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/validation_data.pkl --checkpoint checkpoints/priority_imitation/priority_mse_pairwise_rank.pt --limit 1000 --device cuda --postprocess

# 15. 对官方 check 数据生成第二版优先级模型预测结果
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/check_data_to_students.pkl --checkpoint checkpoints/priority_imitation/priority_mse_pairwise_rank.pt --output outputs/priority_imitation/predictions_priority_mse_pairwise.json --device cuda --postprocess

# 当前默认提交方法仍是 solve.py 的 heuristic
# 只有当 priority_mse_rank.pt 在 validation 上优于默认 heuristic，并且 check 数据运行时间可接受时，再考虑把模型路线作为默认提交方案
# 第二版 priority_mse_pairwise_rank.pt 也必须通过同样判断，不能只看训练 loss

# 16. 强化学习微调 smoke：先检查 RL 训练脚本是否能跑通
python3 scripts/train_priority_rl.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --init-checkpoint checkpoints/priority_imitation/priority_mse_pairwise_rank.pt --checkpoint-output checkpoints/priority_rl/priority_rl_smoke.pt --summary-output outputs/priority_rl/smoke_summary.json --train-limit 16 --eval-limit 8 --epochs 1 --batch-size 8 --samples-per-instance 2 --temperature 1.0 --learning-rate 0.00001 --device cuda --no-postprocess-reward --no-postprocess-eval

# 17. 强化学习正式微调：从 pairwise imitation checkpoint 继续训练
python3 scripts/train_priority_rl.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --init-checkpoint checkpoints/priority_imitation/priority_mse_pairwise_rank.pt --checkpoint-output checkpoints/priority_rl/priority_rl_finetune.pt --summary-output outputs/priority_rl/rl_finetune_summary.json --epochs 20 --batch-size 32 --samples-per-instance 2 --temperature 1.0 --learning-rate 0.00001 --weight-decay 0.0001 --eval-limit 100 --device cuda --postprocess-reward --postprocess-eval

# 18. 在完整 validation 上评估 RL 微调模型
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/validation_data.pkl --checkpoint checkpoints/priority_rl/priority_rl_finetune.pt --limit 1000 --device cuda --postprocess

# 19. 对官方 check 数据生成 RL 预测结果
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/check_data_to_students.pkl --checkpoint checkpoints/priority_rl/priority_rl_finetune.pt --output outputs/priority_rl/predictions_priority_rl.json --device cuda --postprocess
```

## RL 训练完成后复制回当前电脑的文件

第二阶段 RL 正式训练完成后，复制以下文件回当前项目的相同相对路径：

```text
checkpoints/priority_rl/priority_rl_finetune.pt
outputs/priority_rl/rl_finetune_summary.json
outputs/priority_rl/predictions_priority_rl.json
```

如果只完成 smoke，则复制：

```text
checkpoints/priority_rl/priority_rl_smoke.pt
outputs/priority_rl/smoke_summary.json
```
