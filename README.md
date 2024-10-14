# FLCL
Official implement of "Free Lunch: Frame-level Contrastive Learning with Text Perceiver for Robust Scene Text Recognition in Lightweight Models" in PyTorch. This work has been accepted by ***ACM MM 2024***.

### 运行 'train.py' 进行训练
部分与训练相关的配置说明:
- '--train_data'：训练集lmdb文件所在路径;
- '--valid_data'：验证集lmdb文件所在路径;
- '--batch_size'：训练数据的批大小;
- '--CL'：决定是否开启帧级对比学习;
- '--adam'：是否采用Adam进行优化;
- '--lr'：初始学习率;
- '--cycle_lr'：是否开启循环学习率;
- '--max_lr'：开启循环学习率后的最大学习率;
- '--language'：决定训练的语种, 其中'En'表示采用opt.charaters作为训练的标签, 'Zh'表示采用opt.dictionary中的字符作为标签进行训练;
- '--select_data'：训练集文件夹名称;
- '--batch_ratio'：每个训练集中被采样参与训练的数据所占比例;
- '--imgH'：训练图像高度;
- '--imgW'：训练图像宽度;
- '--FeatureExtraction'：特征提取网络(Backbone)型号：
   RCNN | DenseNet | SVTR-T | SVTR-S | EdgeViT-XXS | EdgeViT-XS | EfficientFormerV2-S0 | EfficientFormerV2-S1 | EfficientNet-b0 | EfficientNet-b1 | EfficientNet-b2;
- '--SequenceModeling'：是否采用额外的双向LSTM对上下文特征进行建模, None | BiLSTM;
- '--Prediction'：解码算法 (CTC | Attn), 默认采用CTC解码;

### 运行 'test_lmdb.py' 在lmdb格式的数据集上对模型进行评估
部分相关的配置说明:
- '--eval_data'：评估集路径;
- '--saved_model'：带评估模型参数保存路径;
- '--language'：决定评估的语种, 其中'En'表示采用opt.charaters作为标签, 'Zh'表示采用opt.dictionary中的字符作为标签;
- '--imgH'：评估图像高度;
- '--imgW'：评估图像宽度;
- '--FeatureExtraction'：特征提取网络(Backbone)型号：
   RCNN | DenseNet | SVTR-T | SVTR-S | EdgeViT-XXS | EdgeViT-XS | EfficientFormerV2-S0 | EfficientFormerV2-S1 | EfficientNet-b0 | EfficientNet-b1 | EfficientNet-b2;
- '--SequenceModeling'：决定待评估模型是否采用额外的双向LSTM对上下文特征进行建模, None | BiLSTM;
- '--Prediction'：解码算法 (CTC | Attn), 默认采用CTC解码;

### 运行 'test_figs.py' 在图像/文本对形式的数据集上对模型进行评估
部分相关的配置说明:
- '--savename'：评估结果文件保存名称;
- '--rootpath'：数据集路径;
- '--saved_model'：带评估模型参数保存路径;
- '--language'：决定评估的语种, 其中'En'表示采用opt.charaters作为标签, 'Zh'表示采用opt.dictionary中的字符作为标签;
- '--imgH'：评估图像高度;
- '--imgW'：评估图像宽度;
- '--FeatureExtraction'：特征提取网络(Backbone)型号：
   RCNN | DenseNet | SVTR-T | SVTR-S | EdgeViT-XXS | EdgeViT-XS | EfficientFormerV2-S0 | EfficientFormerV2-S1 | EfficientNet-b0 | EfficientNet-b1 | EfficientNet-b2;
- '--SequenceModeling'：决定待评估模型是否采用额外的双向LSTM对上下文特征进行建模, None | BiLSTM;
- '--Prediction'：解码算法 (CTC | Attn), 默认采用CTC解码;

### 运行 'demo.py' 进行单张文本图像的识别
部分相关的配置说明:
- '--imgpath'：待识别文本图像路径;
- '--saved_model'：带评估模型参数保存路径; 
- '--language'：决定评估的语种, 其中'En'表示采用opt.charaters作为标签, 'Zh'表示采用opt.dictionary中的字符作为标签;
- '--imgH'：评估图像高度;
- '--imgW'：评估图像宽度;
- '--FeatureExtraction'：特征提取网络(Backbone)型号：
   RCNN | DenseNet | SVTR-T | SVTR-S | EdgeViT-XXS | EdgeViT-XS | EfficientFormerV2-S0 | EfficientFormerV2-S1 | EfficientNet-b0 | EfficientNet-b1 | EfficientNet-b2;
- '--SequenceModeling'：决定待评估模型是否采用额外的双向LSTM对上下文特征进行建模, None | BiLSTM;
- '--Prediction'：解码算法 (CTC | Attn), 默认采用CTC解码;

### 预训练模型保存路径：'./saved_models/None-cnnEdgeViT-None-CTC/best_accuracy.pth'
模型配置：--Transformation="None"; --FeatureExtraction="cnnEdgeViT"; --SequenceModeling="None"; --Prediction="CTC"; --output_channel=256
