CNN Demo (Matlab version)
=========================

## Discribe
A handcrafted Convolution Neural Network framwork for DeepLearning study.Very toy but usful when you wanna exactly understand how CNN works and what happen when CNN runing by Debug this program in your Matlab.

* without any dependency
* using pure pictures as input


## configure
Open "CNN_APP.m" and setup parameters list below ,then press "F5" in MATLAB and enjoy yourself:)

* mainPath : usually use '.'
* CNN_Type : lead to training or testing
* netSavePath : where trained Net can be saved or loaded
* pre-process parameters : pre-process your data pictures that use for net input
* training parameters : define Net structure and Hyper parameters for train
* pathStr : Path to your data dir

put your data as below:

	train:Path/to/your/data_dir/train/[ClassIndex]/[pic_name].[format]
	test:Path/to/your/data_dir/test/[ClassIndex]/[pic_name].[format]

[ClassIndex] begins with 1 and [format] can be 'jpg'/'png'/'bmp' and so on.\
For example, mnist picture should be set as follow:

train:
>Path/to/your/data/dir/train/1/1_1.png\
>Path/to/your/data/dir/train/2/2_10.png\
>......

test:
>Path/to/your/data/dir/test/1/1_1.png\
>Path/to/your/data/dir/test/2/2_3.png\
>......\
>Path/to/your/data/dir/test/10/0_1.png\
>......

Mnist(pic version) and its minimun version(just 70 pictures included,60 for train and 10 for test) can be download here:[BaiduYun](http://pan.baidu.com/s/1i4QFyoX). For fast test, I suggest you download Mnist_70(min version), extract it into 'mainPath\Data' and set 'DataPathStr' as '\Data\70\'.

## demo
![](https://github.com/zacchues/CNN_Matlab/blob/master/pic/demo.png)  