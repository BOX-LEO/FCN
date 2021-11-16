## ResNet based FCN

### source list:

|       source        |                         description                          |
| :-----------------: | :----------------------------------------------------------: |
|    **data set**     | [Kitti dataset](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) |
|    **FCN16.py**     |  FCN-16 model, *ResNet18* as pretrain classification model   |
|    **FCN32.py**     |  FCN-32 model, *ResNet18* as pretrain classification model   |
| **KittiDataset.py** |           custom dataset for semantic segmentation           |
|  **experiment.py**  |  1. dataset splitting 2. training and validation processing  |
|     **test.py**     |                         test process                         |
|    **labels.py**    |                   Kitti labels definition                    |



### Setting:

Unlike original FCN setting, both FCN in this report are based on ResNet18

All image are resize to (H: 375, W: 1242) to improve parallelism.

| max_lr | epoch | weight_decay | batch size | n_class |   criterion   | optimizer |
| :----: | :---: | :----------: | :--------: | :-----: | :-----------: | :-------: |
|  1e-3  |  30   |     1e-4     |     10     |   34    | cross entropy |   Adam    |

**Evaluation metrics:** 

1. Pixel-level intersection-over-union (pIoU)
2. Mean Intersection-over-Union (mIoU)



### Training and Validation

1. **FCN 32 loss, mIou, pIoU for training and validation**

![image-20211116113810384](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20211116113810384.png)

2. **FCN 16 loss, mIou, pIoU for training and validation**

![image-20211116114150977](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20211116114150977.png)



### Test Result

|      |                            FCN-16                            |                            FCN-32                            |
| :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| loss |                             1.28                             |                             1.61                             |
| mIoU |                             0.46                             |                             0.34                             |
| pIoU | ![fcn16_test_pIoU](D:\USC\courses\CSCI677\HW\HW5\fcn16_test_pIoU.png) | ![fcn32_test_pIoU](D:\USC\courses\CSCI677\HW\HW5\fcn32_test_pIoU.png) |

| original image   | ![000030_10](D:\USC\courses\CSCI677\HW\HW5\data\image\test\000030_10.png) | **![000045_10](D:\USC\courses\CSCI677\HW\HW5\data\image\test\000045_10.png)** |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Ground Truth** | **![000030_10](D:\USC\courses\CSCI677\HW\HW5\data_semantics\training\semantic_rgb\000030_10.png)** | **![000045_10](D:\USC\courses\CSCI677\HW\HW5\data_semantics\training\semantic_rgb\000045_10.png)** |
| **FCN-32**       | ![image-20211116121716328](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20211116121716328.png) | ![image-20211116121737592](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20211116121737592.png) |
| **FCN-16**       | ![image-20211116121754564](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20211116121754564.png) | ![image-20211116121813593](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20211116121813593.png) |



### Conclusion

FCN32 can only generate coarse segmentation as it use information from deep layer. When fuse it with shallow layer and form FCN16, segmentation are more finer. 



### Reference

[Kaggle](https://www.kaggle.com/ligtfeather/semantic-segmentation-is-easy-with-pytorch/notebook#Training), [github_fcn](https://github.com/wkentaro/pytorch-fcn), [github_layergetter](https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py), [FCN](https://arxiv.org/abs/1605.06211), [ResNet](https://arxiv.org/abs/1512.03385#)

