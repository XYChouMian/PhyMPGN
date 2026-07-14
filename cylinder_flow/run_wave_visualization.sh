#!/bin/bash
# 波方程可视化脚本运行器

echo "=================================================="
echo "波方程可视化程序运行器"
echo "=================================================="

cd /home/wqx/projects/PhyMPGN/cylinder_flow

# 检查conda环境
if ! conda env list | grep -q "phympgn"; then
    echo "错误: phympgn conda环境不存在"
    exit 1
fi

# 选择可视化模式
echo "请选择可视化模式:"
echo "1. 简化版可视化（推荐，无需训练数据）"
echo "2. 完整版可视化（需要训练数据）"
read -p "请输入选择 (1/2): " choice

case $choice in
    1)
        echo "运行简化版可视化..."
        conda run -n phympgn python visualize_wave_simple.py
        ;;
    2)
        echo "运行完整版可视化..."
        conda run -n phympgn python visualize_wave_results.py
        ;;
    *)
        echo "无效选择，运行简化版..."
        conda run -n phympgn python visualize_wave_simple.py
        ;;
esac

echo ""
echo "=================================================="
echo "可视化完成！"
echo "=================================================="

# 显示生成的文件
if [ -d "visualization/wave_simple" ]; then
    echo "简化版可视化结果:"
    ls -lh visualization/wave_simple/*.png
fi

if [ -d "visualization/wave_training_analysis" ]; then
    echo "完整版可视化结果:"
    ls -lh visualization/wave_training_analysis/
fi