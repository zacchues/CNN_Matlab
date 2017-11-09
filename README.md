CNN Demo (Matlab version)
=========================

## Discribe
An easy CNN for study.Very toy but usful when you wanna exactly understand what happen when CNN runing by Debug it in your Matlab.

* without any dependency
* using pure pictures as input


## configure
open "CNN_APP.m" and setup parameters list below ,then press "F5" in MATLAB and enjoy yourself:)

* mainPath:usually set '.'
* CNN_Type:lead to training or testing
* netSavePath:where trained net can be save or load
* pre-process parameters: 
* training parameters: Net structure and training Hyper parameters define
* DataPathStr: Path to your data dir

Keep your data in the following format:

	train:Path/to/your/data_dir/train/[ClassIndex]/[pic_name].[format]
	test:Path/to/your/data_dir/test/[ClassIndex]/[pic_name].[format]

[ClassIndex] begins with 1 and [format] can be 'jpg'/'png'/'bmp' and so on.For example, mnist picture should be set as follow:

train:Path/to/your/data/dir/train/1/1_1.png\<br>  
test:Path/to/your/data/dir/test/10/0_1.png