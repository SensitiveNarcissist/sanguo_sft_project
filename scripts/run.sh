## run.sh

```bash
#!/bin/bash

# 三国演义SFT项目一键运行脚本
# 作者: Your Name
# 日期: 2024年

echo "=========================================="
echo "三国演义SFT监督微调项目 - 一键运行脚本"
echo "=========================================="

# 设置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
MODEL_DIR="$PROJECT_ROOT/models/trained"
DATA_DIR="$PROJECT_ROOT/data/processed"

# 颜色输出函数
print_green() {
    echo -e "\033[32m$1\033[0m"
}

print_red() {
    echo -e "\033[31m$1\033[0m"
}

print_yellow() {
    echo -e "\033[33m$1\033[0m"
}

# 检查目录是否存在
check_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_green "创建目录: $1"
    fi
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_red "错误: 未找到 $1 命令"
        exit 1
    fi
}

# 检查Python依赖
check_python_deps() {
    echo "检查Python依赖..."
    python -c "
import importlib
required = ['torch', 'transformers', 'datasets', 'peft', 'accelerate', 'yaml']
missing = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'缺少依赖: {missing}')
    exit(1)
else:
    print('所有依赖都已安装')
"
    if [ $? -ne 0 ]; then
        print_red "请安装缺失的依赖: pip install -r requirements.txt"
        exit 1
    fi
}

# 步骤1: 环境检查
step1_environment() {
    echo
    print_yellow "步骤1: 检查环境配置..."
    
    # 检查Python
    check_command python
    print_green "✓ Python版本: $(python --version)"
    
    # 检查pip
    check_command pip
    print_green "✓ pip版本: $(pip --version | cut -d' ' -f1-2)"
    
    # 检查Python依赖
    check_python_deps
    
    # 检查CUDA
    if python -c "import torch; print('CUDA可用:', torch.cuda.is_available())" | grep -q "True"; then
        print_green "✓ CUDA可用"
        print_green "✓ GPU设备: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
    else
        print_yellow "⚠ CUDA不可用，将使用CPU训练（速度较慢）"
    fi
    
    # 检查目录结构
    for dir in "$LOG_DIR" "$MODEL_DIR" "$DATA_DIR"; do
        check_directory "$dir"
    done
    
    print_green "环境检查完成！"
}

# 步骤2: 数据准备
step2_data_preparation() {
    echo
    print_yellow "步骤2: 准备训练数据..."
    
    # 检查数据文件是否存在
    if [ -f "$DATA_DIR/sanguo_qa_dataset.json" ]; then
        print_green "✓ 数据文件已存在，跳过数据生成"
        return 0
    fi
    
    # 运行数据生成脚本
    echo "生成三国演义问答数据集..."
    cd "$PROJECT_ROOT"
    
    if [ -f "src/data_augmentation.py" ]; then
        python src/data_augmentation.py
    elif [ -f "scripts/data_augmentation.py" ]; then
        python scripts/data_augmentation.py
    else
        print_red "错误: 未找到数据增强脚本"
        exit 1
    fi
    
    if [ $? -eq 0 ] && [ -f "$DATA_DIR/sanguo_qa_dataset.json" ]; then
        dataset_size=$(python -c "import json; data=json.load(open('$DATA_DIR/sanguo_qa_dataset.json')); print(len(data['samples']))")
        print_green "✓ 数据生成完成！共生成 ${dataset_size} 条问答对"
    else
        print_red "错误: 数据生成失败"
        exit 1
    fi
}

# 步骤3: 模型训练
step3_training() {
    echo
    print_yellow "步骤3: 开始模型训练..."
    
    # 检查是否有已训练的模型
    if [ -f "$MODEL_DIR/sanguo_sft_final/pytorch_model.bin" ] || [ -f "$MODEL_DIR/sanguo_sft_final/adapter_model.bin" ]; then
        echo "检测到已有训练好的模型"
        read -p "是否重新训练？(y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_green "✓ 使用现有模型，跳过训练"
            return 0
        fi
    fi
    
    # 运行训练脚本
    echo "开始SFT训练..."
    cd "$PROJECT_ROOT"
    
    if [ -f "src/train_sft_stable.py" ]; then
        TRAIN_SCRIPT="src/train_sft_stable.py"
    elif [ -f "scripts/train_sft_stable.py" ]; then
        TRAIN_SCRIPT="scripts/train_sft_stable.py"
    else
        print_red "错误: 未找到训练脚本"
        exit 1
    fi
    
    # 设置训练日志
    TRAIN_LOG="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"
    
    echo "训练日志保存到: $TRAIN_LOG"
    python "$TRAIN_SCRIPT" 2>&1 | tee "$TRAIN_LOG"
    
    if [ $? -eq 0 ]; then
        print_green "✓ 模型训练完成！"
        
        # 提取训练结果
        if [ -f "$TRAIN_LOG" ]; then
            echo "训练摘要:"
            grep -E "(训练完成|训练指标|最终训练损失)" "$TRAIN_LOG" | tail -3
        fi
    else
        print_red "错误: 训练失败"
        exit 1
    fi
}

# 步骤4: 模型评估
step4_evaluation() {
    echo
    print_yellow "步骤4: 评估模型性能..."
    
    # 检查模型是否存在
    if [ ! -f "$MODEL_DIR/sanguo_sft_final/pytorch_model.bin" ] && [ ! -f "$MODEL_DIR/sanguo_sft_final/adapter_model.bin" ]; then
        print_red "错误: 未找到训练好的模型"
        return 1
    fi
    
    # 运行评估脚本
    echo "评估模型性能..."
    cd "$PROJECT_ROOT"
    
    if [ -f "src/evaluation.py" ]; then
        EVAL_SCRIPT="src/evaluation.py"
    elif [ -f "scripts/evaluation.py" ]; then
        EVAL_SCRIPT="scripts/evaluation.py"
    else
        print_red "错误: 未找到评估脚本"
        return 1
    fi
    
    # 设置评估日志
    EVAL_LOG="$LOG_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"
    
    echo "评估日志保存到: $EVAL_LOG"
    python "$EVAL_SCRIPT" 2>&1 | tee "$EVAL_LOG"
    
    if [ $? -eq 0 ]; then
        print_green "✓ 模型评估完成！"
        
        # 提取评估结果
        if [ -f "$EVAL_LOG" ]; then
            echo "评估结果:"
            grep -E "(评估完成|精确准确率|按问题类型分析)" "$EVAL_LOG" | tail -5
        fi
    else
        print_yellow "⚠ 评估过程中出现警告，但继续执行"
    fi
}

# 步骤5: 演示推理
step5_demo() {
    echo
    print_yellow "步骤5: 运行推理演示..."
    
    # 检查模型是否存在
    if [ ! -f "$MODEL_DIR/sanguo_sft_final/pytorch_model.bin" ] && [ ! -f "$MODEL_DIR/sanguo_sft_final/adapter_model.bin" ]; then
        print_red "错误: 未找到训练好的模型"
        return 1
    fi
    
    # 演示推理
    echo "启动推理演示..."
    cd "$PROJECT_ROOT"
    
    DEMO_SCRIPT=""
    if [ -f "scripts/inference_demo.py" ]; then
        DEMO_SCRIPT="scripts/inference_demo.py"
    elif [ -f "src/inference_demo.py" ]; then
        DEMO_SCRIPT="src/inference_demo.py"
    fi
    
    if [ -n "$DEMO_SCRIPT" ]; then
        echo "运行演示脚本: $DEMO_SCRIPT"
        echo "按Ctrl+C退出演示"
        python "$DEMO_SCRIPT"
    else
        print_yellow "⚠ 未找到演示脚本，跳过演示"
    fi
}

# 主函数
main() {
    echo "开始执行三国演义SFT项目完整流程..."
    echo "项目根目录: $PROJECT_ROOT"
    echo "当前时间: $(date)"
    echo "=========================================="
    
    # 执行所有步骤
    step1_environment
    step2_data_preparation
    step3_training
    step4_evaluation
    step5_demo
    
    echo
    print_green "=========================================="
    print_green "所有步骤完成！"
    print_green "三国演义SFT项目执行完毕"
    print_green "=========================================="
    
    # 输出关键文件位置
    echo
    echo "关键文件位置:"
    echo "训练好的模型: $MODEL_DIR/sanguo_sft_final/"
    echo "数据集: $DATA_DIR/sanguo_qa_dataset.json"
    echo "训练日志: $LOG_DIR/training_*.log"
    echo "评估日志: $LOG_DIR/evaluation_*.log"
    
    echo
    echo "下一步:"
    echo "1. 使用训练好的模型进行推理"
    echo "2. 调整配置并重新训练以获得更好效果"
    echo "3. 扩展数据集以覆盖更多三国知识"
}

# 处理命令行参数
case "$1" in
    "env"|"environment")
        step1_environment
        ;;
    "data")
        step2_data_preparation
        ;;
    "train")
        step3_training
        ;;
    "eval"|"evaluate")
        step4_evaluation
        ;;
    "demo")
        step5_demo
        ;;
    "all")
        main
        ;;
    "help"|"--help"|"-h")
        echo "用法: $0 [command]"
        echo ""
        echo "可用命令:"
        echo "  all       运行完整流程（默认）"
        echo "  env       仅检查环境"
        echo "  data      仅准备数据"
        echo "  train     仅训练模型"
        echo "  eval      仅评估模型"
        echo "  demo      仅运行演示"
        echo "  help      显示此帮助信息"
        echo ""
        echo "示例:"
        echo "  $0          # 运行完整流程"
        echo "  $0 train    # 仅训练模型"
        ;;
    *)
        # 默认运行完整流程
        main
        ;;
esac

exit 0