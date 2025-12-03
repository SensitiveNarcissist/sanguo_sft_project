#!/usr/bin/env python3
# scripts/train_sft_stable.py

import torch
import yaml
import json
import os
from typing import Dict
import logging
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForSeq2Seq, set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ensure_numeric(value):
    """确保配置值为数值类型"""
    if isinstance(value, str):
        if 'e' in value.lower():
            try:
                return float(value)
            except ValueError:
                pass
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    return value


def load_and_validate_config(config_path: str) -> Dict:
    """加载并验证配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    training_config = config['training_config']
    numeric_fields = [
        'learning_rate', 'warmup_ratio', 'weight_decay',
        'adam_epsilon', 'max_grad_norm', 'per_device_train_batch_size',
        'per_device_eval_batch_size', 'gradient_accumulation_steps',
        'num_train_epochs', 'logging_steps', 'eval_steps', 'save_steps'
    ]

    for field in numeric_fields:
        if field in training_config:
            training_config[field] = ensure_numeric(training_config[field])

    # 确保 save_steps 是 eval_steps 的整数倍
    eval_steps = training_config['eval_steps']
    save_steps = training_config['save_steps']

    if save_steps % eval_steps != 0:
        new_save_steps = (save_steps // eval_steps) * eval_steps
        if new_save_steps == 0:
            new_save_steps = eval_steps
        logger.warning("自动调整 save_steps 从 %s 到 %s (eval_steps 的整数倍)",
                       save_steps, new_save_steps)
        training_config['save_steps'] = new_save_steps

    return config


def setup_model_and_tokenizer(config):
    """设置模型和分词器 - 稳定版本"""
    model_config = config['model_config']

    logger.info("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['model_name'],
        trust_remote_code=model_config['trust_remote_code'],
        local_files_only=model_config.get('local_files_only', False)
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("设置pad_token为eos_token")

    logger.info("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config['model_name'],
        torch_dtype=getattr(torch, model_config['torch_dtype']),
        device_map="auto",
        trust_remote_code=model_config['trust_remote_code'],
        local_files_only=model_config.get('local_files_only', False)
    )

    # 确保模型在训练模式
    model.train()
    logger.info("模型设置为训练模式")

    # 应用LoRA
    optimization_config = config['optimization_config']
    if optimization_config['use_lora']:
        logger.info("应用LoRA配置...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=optimization_config['lora_rank'],
            lora_alpha=optimization_config['lora_alpha'],
            lora_dropout=optimization_config['lora_dropout'],
            target_modules=optimization_config['lora_target_modules'],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        # 详细检查可训练参数
        trainable_params = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                logger.debug("可训练参数: %s (形状: %s)", name, param.shape)

        logger.info("可训练参数: %s / %s (%.4f%%)",
                    f"{trainable_params:,}", f"{total_params:,}",
                    100 * trainable_params / total_params)

        # 使用Peft的打印功能
        model.print_trainable_parameters()

    # 注意：暂时禁用梯度检查点，因为它可能导致梯度问题
    if optimization_config.get('use_gradient_checkpointing', False):
        logger.warning("梯度检查点已禁用，以避免梯度问题")
        # model.gradient_checkpointing_enable()  # 暂时注释掉

    # 验证模型参数状态
    logger.info("验证模型参数状态...")
    has_trainable = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_trainable = True
            break

    if not has_trainable:
        logger.error("错误：没有可训练的参数！")
        raise RuntimeError("模型没有可训练参数，请检查LoRA配置")

    logger.info("模型设置完成，有可训练参数")
    return model, tokenizer


def create_training_args(config, path_config):
    """创建训练参数 - 稳定版本"""
    training_config = config['training_config']

    base_args = {
        'output_dir': path_config['output_dir'],
        'overwrite_output_dir': True,

        # 批次配置
        'per_device_train_batch_size': training_config['per_device_train_batch_size'],
        'per_device_eval_batch_size': training_config['per_device_eval_batch_size'],
        'gradient_accumulation_steps': training_config['gradient_accumulation_steps'],

        # 优化器配置
        'learning_rate': training_config['learning_rate'],
        'warmup_ratio': training_config['warmup_ratio'],
        'weight_decay': training_config['weight_decay'],
        'adam_epsilon': training_config['adam_epsilon'],
        'max_grad_norm': training_config['max_grad_norm'],

        # 训练周期
        'num_train_epochs': training_config['num_train_epochs'],
        'max_steps': training_config['max_steps'],

        # 日志和评估
        'logging_dir': path_config['logging_dir'],
        'logging_steps': training_config['logging_steps'],
        'eval_steps': training_config['eval_steps'],
        'save_steps': training_config['save_steps'],

        # 模型保存
        'load_best_model_at_end': training_config['load_best_model_at_end'],
        'metric_for_best_model': training_config['metric_for_best_model'],
        'greater_is_better': training_config['greater_is_better'],
        'save_total_limit': training_config['save_total_limit'],

        # 硬件优化
        'fp16': training_config['fp16'],
        'dataloader_pin_memory': training_config['dataloader_pin_memory'],
        'remove_unused_columns': training_config['remove_unused_columns'],

        # 报告设置
        'report_to': training_config['report_to'],
        'run_name': "sanguo_sft_stable"
    }

    # 处理版本兼容性
    try:
        # 尝试新版本参数名
        test_args = TrainingArguments(output_dir="./test", eval_strategy="steps")
        base_args['eval_strategy'] = training_config['evaluation_strategy']
        base_args['save_strategy'] = training_config['save_strategy']
        logger.info("使用新版本参数名 (eval_strategy)")
    except TypeError:
        try:
            # 尝试旧版本参数名
            test_args = TrainingArguments(output_dir="./test", evaluation_strategy="steps")
            base_args['evaluation_strategy'] = training_config['evaluation_strategy']
            base_args['save_strategy'] = training_config['save_strategy']
            logger.info("使用旧版本参数名 (evaluation_strategy)")
        except Exception:
            # 使用默认值
            base_args['eval_strategy'] = "steps"
            base_args['save_strategy'] = "steps"
            logger.info("使用默认评估策略")

    logger.info("稳定训练参数:")
    logger.info("  学习率: %.2e", base_args['learning_rate'])
    logger.info("  批次大小: %s", base_args['per_device_train_batch_size'])
    logger.info("  梯度累积: %s", base_args['gradient_accumulation_steps'])

    return TrainingArguments(**base_args)


def prepare_dataset(config, tokenizer):
    """准备数据集 - 稳定版本"""
    logger.info("加载数据集...")
    try:
        with open("data/processed/sanguo_qa_dataset.json", 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        logger.error("数据文件不存在，请先运行 data_collection.py 和 data_augmentation.py")
        return None

    # 使用适量数据确保稳定性
    all_samples = raw_data["samples"]
    sample_size = min(500, len(all_samples))  # 使用500个样本确保稳定性
    formatted_data = []

    logger.info("使用 %s 个样本进行训练", sample_size)

    for i, sample in enumerate(all_samples[:sample_size]):
        prompt = config['data_config']['prompt_template'].format(question=sample['question'])
        formatted_data.append({
            "input": prompt,
            "output": sample['answer'],
            "id": i
        })

    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.train_test_split(
        test_size=1 - config['data_config']['train_test_split'],
        seed=config['data_config']['random_seed']
    )

    logger.info("稳定数据集: %s 训练样本, %s 验证样本",
                len(dataset['train']), len(dataset['test']))

    # 分词函数
    def tokenize_function(examples):
        texts = []
        for input_text, output_text in zip(examples['input'], examples['output']):
            full_text = input_text + output_text + tokenizer.eos_token
            texts.append(full_text)

        tokenized = tokenizer(
            texts,
            max_length=config['model_config']['max_length'],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        labels = tokenized["input_ids"].clone()
        for i, text in enumerate(texts):
            input_text = examples['input'][i]
            input_ids = tokenizer.encode(input_text, add_special_tokens=False)
            if len(input_ids) < len(labels[i]):
                labels[i, :len(input_ids)] = -100

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

    # 分词处理
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=1
    )

    return tokenized_datasets


def test_model_gradients(model, tokenizer):
    """测试模型梯度功能"""
    logger.info("测试模型梯度...")

    try:
        # 创建测试输入
        test_input = "测试输入"
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
        labels = inputs["input_ids"].clone()

        # 前向传播
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # 检查损失是否需要梯度
        if not loss.requires_grad:
            logger.error("损失张量不需要梯度！")
            return False

        # 反向传播测试
        loss.backward()

        # 检查是否有参数获得梯度
        has_gradients = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                logger.info("参数 %s 获得梯度", name)
                break

        if has_gradients:
            logger.info("✅ 模型梯度测试通过")
            return True
        else:
            logger.error("❌ 没有参数获得梯度")
            return False

    except Exception as e:
        logger.error("梯度测试失败: %s", e)
        return False


def main():
    try:
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 加载并验证配置
        config = load_and_validate_config("config/training_config.yaml")

        # 设置种子
        set_seed(config['data_config']['random_seed'])

        # 创建目录
        os.makedirs(config['path_config']['output_dir'], exist_ok=True)
        os.makedirs(config['path_config']['logging_dir'], exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # 设置模型和分词器
        model, tokenizer = setup_model_and_tokenizer(config)

        # 测试模型梯度
        if not test_model_gradients(model, tokenizer):
            logger.error("模型梯度测试失败，停止训练")
            return

        # 准备数据集
        tokenized_datasets = prepare_dataset(config, tokenizer)
        if tokenized_datasets is None:
            return

        # 创建训练参数
        training_args = create_training_args(config, config['path_config'])

        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )

        # 创建Trainer - 使用兼容的参数名
        trainer_kwargs = {
            'model': model,
            'args': training_args,
            'train_dataset': tokenized_datasets["train"],
            'eval_dataset': tokenized_datasets["test"],
            'data_collator': data_collator,
        }

        # 根据transformers版本使用正确的参数名
        try:
            # 尝试新版本参数名
            trainer_kwargs['tokenizer'] = tokenizer
        except TypeError:
            # 使用旧版本参数名
            trainer_kwargs['processing_class'] = tokenizer

        trainer = Trainer(**trainer_kwargs)

        # 开始训练
        logger.info("开始稳定训练...")
        train_result = trainer.train()

        # 保存模型
        final_model_dir = config['path_config']['final_model_dir']
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        logger.info("稳定训练完成！模型已保存到: %s", final_model_dir)

        # 保存训练指标
        metrics = train_result.metrics
        logger.info("训练指标: %s", metrics)

        # 分析训练效果
        analyze_training_results(metrics)

    except Exception as e:
        logger.error("训练失败: %s", e)
        import traceback
        logger.error("详细错误信息: %s", traceback.format_exc())
        raise


def analyze_training_results(metrics):
    """分析训练结果"""
    logger.info("训练结果分析:")
    final_loss = metrics.get('train_loss', 0)
    logger.info("最终训练损失: %.4f", final_loss)

    if 'eval_loss' in metrics:
        logger.info("最终评估损失: %.4f", metrics['eval_loss'])

    # 给出改进建议
    if final_loss > 6.0:
        logger.info("改进建议:")
        logger.info("  1. 考虑增加训练数据量")
        logger.info("  2. 调整学习率到3e-5")
        logger.info("  3. 增加训练轮数")
    elif final_loss > 4.0:
        logger.info("训练效果中等，可以继续优化")
    else:
        logger.info("训练效果良好！")


if __name__ == "__main__":
    main()