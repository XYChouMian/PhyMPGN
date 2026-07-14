# conda activate phympgn
python generate_data_wave.py --file configs/train_wave.yaml  # 生成训练/验证数据缓存
python train_wave.py --file configs/train_wave.yaml          # 开始训练
