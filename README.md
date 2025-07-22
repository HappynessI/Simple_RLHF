# Simple RLHF Tiny 🚀

一个简化的强化学习人类反馈（RLHF）实现，专为学习和研究目的设计。本项目提供了完整的 RLHF 训练流程，包括 Actor 模型、Critic 模型的训练以及最终的人类反馈强化学习。

## ✨ 功能特性

- 🎭 **Actor 模型训练**：基于 Transformers 的语言模型微调
- 🎯 **Critic 模型训练**：奖励模型训练，用于评估生成文本质量
- 🔄 **RLHF 训练**：使用 PPO 算法进行人类反馈强化学习
- 🧪 **模型测试**：完整的推理测试和结果评估
- ☁️ **Google Colab 支持**：开箱即用的云端运行环境
- 📊 **数据集支持**：支持自定义数据集格式
- 🛠️ **模块化设计**：清晰的代码结构，易于理解和修改

## 📋 目录

- [安装要求](#安装要求)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [训练流程](#训练流程)
- [Google Colab 使用](#google-colab-使用)
- [数据集格式](#数据集格式)
- [模型配置](#模型配置)
- [故障排除](#故障排除)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 🔧 安装要求

### 基础环境
- Python 3.8+
- CUDA 11.0+（可选，用于 GPU 加速）

### 依赖包
```bash
pip install torch transformers accelerate datasets
pip install numpy pandas tqdm jupyter
```

### 克隆项目
```bash
git clone https://github.com/your-username/Simple_RLHF_tiny.git
cd Simple_RLHF_tiny
```

## 🚀 快速开始

### 本地运行

1. **准备数据集**
   ```bash
   # 确保 dataset/ 目录包含训练数据
   ls dataset/
   # 应该看到: train.json eval.json
   ```

2. **下载预训练模型**
   ```bash
   jupyter notebook "0.下载文件.ipynb"
   ```

3. **训练 Actor 模型**
   ```bash
   jupyter notebook "1.actor.ipynb"
   ```

4. **训练 Critic 模型**
   ```bash
   jupyter notebook "2.critic.ipynb"
   ```

5. **RLHF 训练**
   ```bash
   jupyter notebook "3.rlhf.ipynb"
   ```

6. **模型测试**
   ```bash
   jupyter notebook "4.test.ipynb"
   ```

### Google Colab 运行

1. **上传项目到 Google Drive**
2. **打开 Colab 并运行**：
   - 建议使用 GPU 运行时
   - 按顺序执行各个 notebook

## 📁 项目结构

```
Simple_RLHF_tiny/
├── 📓 0.下载文件.ipynb      # 模型和数据准备
├── 📓 1.actor.ipynb         # Actor 模型训练
├── 📓 2.critic.ipynb        # Critic 模型训练  
├── 📓 3.rlhf.ipynb          # RLHF 强化学习训练
├── 📓 4.test.ipynb          # 模型测试和评估
├── 🐍 util.py               # 工具函数和类
├── 📊 dataset/              # 训练数据集
│   ├── train.json           # 训练数据
│   └── eval.json            # 评估数据
├── 🤖 model/                # 存放训练的模型
│   └── rlhf/                # RLHF 训练后的模型
├── 🔤 tokenizer/            # 分词器文件
└── 📖 README.md             # 项目说明文档
```

## 🔄 训练流程

### 1. 数据准备阶段
- 下载预训练模型（如 `facebook/opt-125m`）
- 准备训练数据集（JSON 格式）
- 初始化分词器

### 2. Actor 模型训练
- 基于预训练模型进行监督微调（SFT）
- 训练目标：学习生成高质量的响应
- 输出：Actor 模型（用于生成文本）

### 3. Critic 模型训练
- 训练奖励模型（Reward Model）
- 学习评估文本质量和人类偏好
- 输出：Critic 模型（用于评分）

### 4. RLHF 训练
- 使用 PPO（Proximal Policy Optimization）算法
- Actor 生成文本，Critic 提供奖励
- 通过强化学习优化模型表现

### 5. 模型评估
- 在测试集上评估模型性能
- 对比训练前后的改进效果

## ☁️ Google Colab 使用

### 环境配置
1. **设置 GPU 运行时**：运行时 → 更改运行时类型 → GPU
2. **上传项目文件**：
   - 方法一：直接拖拽到 Colab 文件区
   - 方法二：上传到 Google Drive 后挂载

### 特殊说明
- 🔧 自动检测项目文件位置
- 💾 自动安装所需依赖
- 🛡️ 内置错误处理和备用方案
- 🧹 自动清理 GPU 缓存

### 故障排除
- **模型加载失败**：自动使用备用模型
- **文件缺失**：提供详细上传指导
- **内存不足**：自动调整参数或切换到 CPU

## 📊 数据集格式

### 训练数据格式 (train.json)
```json
{
  "prompt": "Human: 问题描述 Assistant:",
  "chosen": "好的回答",
  "rejected": "差的回答", 
  "response": "模型应该学习的回答"
}
```

### 评估数据格式 (eval.json)
```json
{
  "prompt": "Human: 测试问题 Assistant:",
  "chosen": "期望的答案",
  "rejected": "",
  "response": "用于对比的答案"
}
```

### 自定义数据集
1. 准备符合格式的 JSON 文件
2. 放置在 `dataset/` 目录下
3. 更新 notebook 中的数据加载路径

## ⚙️ 模型配置

### 支持的基础模型
- `facebook/opt-125m`（默认，适合快速测试）
- `facebook/opt-350m`（更好的性能）
- `microsoft/DialoGPT-small`
- 其他兼容的 Transformers 模型

### 训练参数
```python
# Actor 训练参数
learning_rate = 2e-5
batch_size = 4
num_epochs = 3
max_length = 512

# Critic 训练参数  
learning_rate = 1e-5
batch_size = 8
num_epochs = 1

# RLHF 参数
ppo_epochs = 4
kl_coef = 0.1
clip_ratio = 0.2
```

## 🔍 故障排除

### 常见问题

**Q: CUDA 内存不足怎么办？**
```python
# 减少批次大小
batch_size = 2

# 使用梯度累积
gradient_accumulation_steps = 4

# 启用梯度检查点
gradient_checkpointing = True
```

**Q: 模型训练速度慢怎么办？**
- 使用更小的模型（如 opt-125m）
- 减少序列长度
- 启用混合精度训练

**Q: 生成结果质量不好怎么办？**
- 增加训练轮数
- 调整学习率
- 使用更好的基础模型
- 改善训练数据质量

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

2. **检查数据加载**
   ```python
   # 验证数据格式
   with open('dataset/train.json') as f:
       sample = json.loads(f.readline())
       print(sample)
   ```

3. **监控训练进度**
   - 使用 TensorBoard 或 wandb
   - 定期保存检查点
   - 在验证集上评估

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 贡献类型
- 🐛 Bug 修复
- ✨ 新功能开发
- 📚 文档改进
- 🧪 测试用例
- 🎨 代码优化

### 开发规范
- 遵循 PEP 8 代码风格
- 添加适当的注释和文档
- 编写测试用例
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 预训练模型和工具
- [Facebook OPT](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT) - 基础语言模型
- [OpenAI](https://openai.com/) - RLHF 方法论
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 大规模训练优化

## 📞 联系方式

- **项目链接**：[https://github.com/your-username/Simple_RLHF_tiny](https://github.com/your-username/Simple_RLHF_tiny)
- **问题反馈**：[Issues](https://github.com/your-username/Simple_RLHF_tiny/issues)
- **讨论交流**：[Discussions](https://github.com/your-username/Simple_RLHF_tiny/discussions)

## 🌟 Star History

如果这个项目对您有帮助，请给我们一个 ⭐ Star！

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/Simple_RLHF_tiny&type=Date)](https://star-history.com/#your-username/Simple_RLHF_tiny&Date)

---

**Happy Coding! 🎉**
