# conda activate phympgn
python generate_data_BL.py --file configs/train_BL.yaml  # 生成训练/验证数据缓存
python train_BL.py --file configs/train_BL.yaml          # 开始训练
