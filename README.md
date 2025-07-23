# Simple RLHF - 人类反馈强化学习

一个轻量、快速、简单的RLHF（Reinforcement Learning with Human Feedback）实现。

## 项目简介

本项目提供了一个完整的RLHF训练流程实现，包括Actor模型训练、Critic模型训练、RLHF微调和测试评估。

## 项目结构

```
Simple_RLHF_tiny-main/
├── 1.actor.ipynb          # Actor模型训练
├── 2.critic.ipynb         # Critic模型训练  
├── 3.rlhf.ipynb          # RLHF强化学习训练
├── 4.test.ipynb          # 模型测试和评估
├── dataset/              # 数据集
│   ├── train.json        # 训练数据
│   └── eval.json         # 评估数据
├── model/                # 模型存储
│   ├── actor/            # Actor模型文件
│   ├── critic/           # Critic模型文件
│   └── rlhf/            # RLHF训练后的模型
├── tokenizer/            # 分词器
│   └── facebook/opt-350m/
├── util.py              # 工具函数
└── README.md            # 项目说明
```

## 环境要求

```bash
torch==1.13.1+cu117
transformers==4.38.2
datasets==2.18.0
accelerate==0.28.0
peft==0.9.0
```

## 使用说明

按照以下顺序运行notebook文件：

1. **1.actor.ipynb** - 训练Actor模型（生成模型）
2. **2.critic.ipynb** - 训练Critic模型（奖励模型）
3. **3.rlhf.ipynb** - 使用PPO算法进行RLHF训练
4. **4.test.ipynb** - 测试训练后的模型效果

## 快速开始

1. 安装依赖环境
2. 准备训练数据（已包含在dataset目录中）
3. 按顺序运行notebook文件
4. 查看训练结果和模型效果

## 参考资料

- 原版代码: https://github.com/lansinuote/Simple_RLHF
- 视频课程: https://www.bilibili.com/video/BV13r42177Hk

## 说明

这是一个教学目的的RLHF实现，适合学习和理解RLHF的基本原理和实现过程。
