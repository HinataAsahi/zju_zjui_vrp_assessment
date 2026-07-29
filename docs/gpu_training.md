# GPU 迁移运行指令

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
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions.json --device cuda

# 9. 用很小规模 smoke 训练检查训练脚本、数据读取、checkpoint 写入是否正常
python3 scripts/train_priority_model.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --checkpoint-output checkpoints/priority_smoke.pt --summary-output outputs/priority_smoke_summary.json --train-limit 16 --eval-limit 8 --epochs 1 --batch-size 8 --hidden-dim 64 --num-heads 4 --num-layers 1 --dropout 0.1 --device cuda --no-postprocess-eval

# 10. 正式训练第一版客户优先级模型；输出 best checkpoint 和训练摘要
python3 scripts/train_priority_model.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --checkpoint-output checkpoints/priority_mse_rank.pt --summary-output outputs/priority_mse_rank_summary.json --epochs 50 --batch-size 64 --hidden-dim 128 --num-heads 4 --num-layers 2 --dropout 0.1 --learning-rate 0.001 --weight-decay 0.0001 --eval-limit 100 --device cuda --postprocess-eval

# 11. 在完整 validation 上评估训练出的优先级模型
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/validation_data.pkl --checkpoint checkpoints/priority_mse_rank.pt --limit 1000 --device cuda --postprocess

# 12. 对官方 check 数据生成优先级模型预测结果，便于和 heuristic 输出对比
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/check_data_to_students.pkl --checkpoint checkpoints/priority_mse_rank.pt --output outputs/predictions_priority_model.json --device cuda --postprocess

# 13. 正式训练第二版客户优先级模型：MSE + pairwise ranking loss
python3 scripts/train_priority_model.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --checkpoint-output checkpoints/priority_mse_pairwise_rank.pt --summary-output outputs/priority_mse_pairwise_rank_summary.json --epochs 50 --batch-size 64 --hidden-dim 128 --num-heads 4 --num-layers 2 --dropout 0.1 --learning-rate 0.001 --weight-decay 0.0001 --eval-limit 100 --device cuda --postprocess-eval --loss mse_pairwise --pairwise-weight 0.5 --pairwise-margin 0.1

# 14. 在完整 validation 上评估第二版优先级模型
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/validation_data.pkl --checkpoint checkpoints/priority_mse_pairwise_rank.pt --limit 1000 --device cuda --postprocess

# 15. 对官方 check 数据生成第二版优先级模型预测结果
python3 scripts/evaluate_priority_model.py --input VRP_project/VRPData/check_data_to_students.pkl --checkpoint checkpoints/priority_mse_pairwise_rank.pt --output outputs/predictions_priority_mse_pairwise.json --device cuda --postprocess

# 当前默认提交方法仍是 solve.py 的 heuristic
# 只有当 priority_mse_rank.pt 在 validation 上优于默认 heuristic，并且 check 数据运行时间可接受时，再考虑把模型路线作为默认提交方案
# 第二版 priority_mse_pairwise_rank.pt 也必须通过同样判断，不能只看训练 loss
```
