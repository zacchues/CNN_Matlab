CNN Demo (Matlab version)
=========================

## Discribe
An easy CNN for study.Very toy but usful when you wanna exactly understand what happen when CNN runing by Debug it in your Matlab.

* without any dependency
* using pure pictures as input


## configure
open "CNN_APP.m" and setup parameters list below ,then press "F5" in MATLAB and enjoy yourself:)

* mainPath : usually set '.'
* CNN_Type : lead to training or testing
* netSavePath : where trained Net can be saved or loaded
* pre-process parameters: 
* training parameters :define Net structure and training Hyper parameters 
* DataPathStr : Path to your data dir

put your data as below:

	train:Path/to/your/data_dir/train/[ClassIndex]/[pic_name].[format]
	test:Path/to/your/data_dir/test/[ClassIndex]/[pic_name].[format]

[ClassIndex] begins with 1 and [format] can be 'jpg'/'png'/'bmp' and so on.For example, mnist picture should be set as follow:

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
