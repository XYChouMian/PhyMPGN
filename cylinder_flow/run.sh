# conda activate phympgn
python generate_data.py --file configs/train.yaml  # 生成训练/验证数据缓存
python train.py --file configs/train.yaml          # 开始训练