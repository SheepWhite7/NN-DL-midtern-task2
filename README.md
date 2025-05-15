# MMDetection框架下基于VOC和COCo数据集的实例分割、目标检测训练与实践
本教程详细介绍如何在MMDetection框架下，使用Pytorch环境对VOC数据集进行目标检测任务的训练、测试及结果可视化。

## 一、创建Pytorch环境
1. 使用conda创建名为`openmmlab`的环境，并指定Python版本为3.8：
```bash
conda create --name openmmlab python=3.8 -y
```
2. 激活创建好的环境：
```bash
conda activate openmmlab
```
3. 安装Pytorch和Torchvision：
```bash
conda install pytorch torchvision -c pytorch
```

## 二、下载MMDetection框架
1. 安装`openmim`并更新到最新版本：
```bash
pip install -U openmim
```
2. 使用`openmim`安装`mmengine`和`mmcv`：
```bash
mim install mmengine
mim install "mmcv>=2.0.0"
```
3. 克隆MMDetection仓库到本地：
```bash
git clone https://github.com/open-mmlab/mmdetection.git  
cd mmdetection
```
4. 在可编辑模式下安装MMDetection：
```bash
pip install -v -e .
```
注：`-v`表示详细输出，`-e`表示可编辑模式安装，方便后续代码修改生效。

## 三、验证安装
1. 下载预训练模型配置文件：
```bash
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```
2. 使用示例图片进行目标检测演示：
```bash
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```
检测结果会保存在当前文件夹下的`outputs/vis`文件夹中，名为`demo.jpg`，图片包含网络预测的检测框。

## 四、数据准备
1. **下载数据**
    - 下载VOC2012数据集：
    ```bash
    wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar  
    tar -xvf VOCtrainval_11-May-2012.tar
    ```
    - 下载COCO数据集：
    ```bash
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    unzip train2017.zip
    unzip val2017.zip
    unzip annotations_trainval2017.zip
    ```
2. （如果是VOC数据集）**转化为COCO格式**
切换到`mmdetection`目录，执行数据格式转换脚本：
```bash
cd mmdetection
python /mmdetection/data/data_process.py
```
3. （如果是VOC数据集）**提取实例分割标注信息**
执行实例分割标注提取脚本：
```bash
python /mmdetection/data/segment_extract.py
```
4. **修改配置文件**
    - `/mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py`：将`num_classes`修改为20（需修改两处）。
    - `/mmdetection/mmdet/evaluation/functional/class_names.py`：修改`coco_classes()`中的类别信息。
    - `/mmdetection/mmdet/datasets/coco.py`：修改`CocoDataset`中的类别及对应颜色。
    - `/mmdetection/configs/_base_/default_runtime.py`：在`vis_backends`中添加`dict(type='TensorboardVisBackend')`，用于后续查看tensorboard。
    - `/mmdetection/configs/_base_/schedules/schedule_1x.py`：若需要调整训练参数，如`epoch`、`lr`等，可在此文件中修改。
5. **开始训练**
    - **Mask-RCNN训练**：
    ```bash
    torchrun --nproc_per_node=4 tools/train.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py --work-dir ./work_dirs/mask-rcnn_r50_fpn_1x_coco --launcher pytorch
    ```
    - **Sparse-RCNN训练**：
    ```bash
    torchrun --nproc_per_node=4 tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py --work-dir ./work_dirs/sparse-rcnn_r50_fpn_1x_coco --launcher pytorch
    ```
6. **查看tensorboard**
以Mask-RCNN为例（Sparse-RCNN只需更改配置文件路径），启动tensorboard：
```bash
tensorboard --logdir='/mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_coco' --host=127.0.0.1 --port=8008
```
7. **测试结果可视化**
```bash
python tools/test.py /mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_coco/mask-rcnn_r50_fpn_1x_coco.py /mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_12.pth --show-dir vis_results/
```
8. **查看Mask-RCNN第一阶段的proposal box**
```bash
python tools/test.py /mmdetection/configs/rpn/rpn_r50_fpn_1x_coco.py  /mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_12.pth  --show-dir ./output_rpn
```
9. **使用保存好的checkpoint可视化自己的图片**
```bash
python demo/image_demo.py /mmdetection/data/my_image/img1.jpg /mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py     --weights /mnt/tidal-alsh01/usr/yangshiyue/mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_coco_real/epoch_12.pth     --out-dir ./output_my_image
```
