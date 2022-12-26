## HybridNets目标检测与图像分割融合模型 –Pytorch实现
---

## 目录  
1. [所需环境 Environment](#所需环境) 
2. [注意事项 Attention](#注意事项) 
3. [模型结构 Network Structure](#模型结构)
4. [效果展示 Effect](#效果展示)
5. [数据下载 Download](#数据下载) 
6. [训练步骤 Train](#训练步骤) 
7. [预测步骤 Predict](#预测步骤) 
8. [参考文献 Reference](#参考文献) 

## 所需环境  
1. Python3.7
2. Pytorch>=1.10.1+cu113  
3. Torchvision>=0.11.2+cu113
4. timm>=0.6.11
5. Tensorflow>=2.5.0(非必须)
6. Tensorflow-gpu>=2.5.0(非必须)
7. Numpy==1.19.5
8. Pillow==8.2.0
9. Opencv-contrib-python==4.5.1.48
10. onnx==1.12.0
11. onnx-tf==1.10.0(非必须)
12. onnxruntime==1.12.1
13. onnxruntime-gpu==1.12.1
14. CUDA 11.0+
15. Cudnn 8.0.4+
16. Docker(非必须)

## 注意事项  
1. 实现基于effecientnet骨干的HybridNets，用于检测目标，同时分割关键区域
2．真实框与先验框的标签整定使用借鉴Retina/RetinaFace，https://github.com/biubug6/Pytorch_Retinaface 
3. 借鉴RetinaFace的检测体置信度、坐标位置误差计算方法
4. 图像分割误差直接使用BCE误差，为避免过拟合可更换为Focal误差
5. 加入正则化操作，降低过拟合影响  
6. 数据与标签路径、训练参数等均位于config.py  
7. onnx通用部署模型转换位于
```python
./onnx
```
8. tensorflow pb模型需执行  
```python
onnx2pb.py
```
9. 产出pb模型，tensorflow serving部署指令：
```python
docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=根目录/pb_model/hybridnet,target=/models/hybridnet -e MODEL_NAME= hybridnet -t tensorflow/serving:版本
```
10. 本项目提供的pb模型较弱，训练次数较少，仅供参考

## 模型结构  
![image](https://github.com/JJASMINE22/HybridNets/blob/main/structure/hybridnets.jpg)  

## 效果展示
![image](https://github.com/JJASMINE22/HybridNets/blob/main/results/sample1.jpg)  
![image](https://github.com/JJASMINE22/HybridNets/blob/main/results/sample2.jpg)  

## 数据下载    
BDD100K  
链接：https://bdd-data.berkeley.edu/portal.html#download
下载解压后将数据集放置于config.py中指定的路径。 

## 训练步骤  
运行train.py  

## 预测步骤  
运行predict.py  

## 参考文献  
https://arxiv.org/abs/2203.09035  
