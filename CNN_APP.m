function CNN_APP
%卷积神经网络训练主程序

mainPath='E:\Coding\Matlab\CNN';% !!!运行程序前请先设置好主目录路径
addpath(genpath(mainPath));

% 设定当前网络是用来预测还是用来训练
CNN_Type='test';% 预测
% CNN_Type='train';% 训练

% 设定网络存储位置
% netSavePath=[mainPath,'\cnn_Save\test.mat'];
netSavePath=[mainPath,'\cnn_Save\Mnist_cnn.mat'];% 网络存储位置
% netSavePath=[mainPath,'\cnn_Save\Mnist_cnn1.mat'];% 网络存储位置
% netSavePath=[mainPath,'\cnn_Save\20161124_Mushrooms_All_cnn.mat'];% 网络存储位置
% netSavePath=[mainPath,'\cnn_Save\20170421_Mushrooms_All_4class.mat'];% 网络存储位置

% 预处理参数
imgSize=32;% 图像归一化尺寸
isDoGray=0;% 是否做灰度化
isDoBW=0;% 是否做二值化
isDoColorReversal=0;% 是否做颜色反转

if strcmp(CNN_Type,'test')% 预测
    
    CNN_Test(netSavePath,imgSize,isDoGray,isDoBW,isDoColorReversal);
    
elseif strcmp(CNN_Type,'train')% 训练
    %% 设置基本参数
    
    % inputSize=size(input,2);% 输入一维向量维度
    
    batchSize=10;% 训练集的一批batch中包含多少条样本
    
    % 选择代价函数
    costFunction=@ MSE;
    % costFunction=@ softmax_Loss;
    
    % 设置正则化方式。不采用正则化：'nan'，L1正则化：'L1',L2正则化：'L2'
    % 三种方式默认都限制权值大小
    regularizationType='L1';
    regularizationLamda=0.1;% 正则化参数
    
    maxIterNum=10000;% 训练迭代次数
    accuracy=2;% 进度显示精度（保留几位小数点位数）
    isShowProgress=1;% 是否显示实时进度条和误差曲线，1为显示，0为不显示
    isLoadExistCNN=0;% 是否载入现有网络,0表示新建网络（不载入）
    
    
    % 设置输出维度（分类数量）
    outputSize=10;
    
    % ------载入图像数据库------
    
    % 图像数据路径
    % DataPathStr='\Data\scriptNumImgData\training\';
    DataPathStr='\Data\Mnist\70\';
    % DataPathStr='\Data\Mnist\1400\';
%     DataPathStr='\Data\Mnist\';
    % DataPathStr='\Data\Mushrooms_part\';
    % DataPathStr='\Data\Mushrooms_All\';
    % DataPathStr='\Data\Mushrooms_All_plus\';
    % DataPathStr='\Data\Mushrooms_All_plus1\';
    % DataPathStr='\Data\Mushrooms_All_twoPart\';
    % DataPathStr='\Data\Mushrooms_All_rot90_6class\';
    % DataPathStr='\Data\Mnist\all\';
    % DataPathStr='\Data\ORL_train_test\';
    % DataPathStr='\Data\ORL_train_test_(1-4)\';
    % DataPathStr='\Data\Mushrooms_All_centralization_rot360\';
    % DataPathStr='\Data\Mushrooms_All_centralization_rot90\';
    % DataPathStr='\Data\Mushrooms_All_centralization_balance\';
    % DataPathStr='\Data\Mushrooms_All_centralization_2part_(1,2,3,5,6),(4)\';
    %     DataPathStr='\Data\Mushrooms_All_centralization_2part_(1,6),(2,3,5)\';
    %     DataPathStr='\Data\Mushrooms_All_centralization_2part_(2,3),(5)\';
    %     DataPathStr='\Data\Mushrooms_All_centralization_2part_(1),(6)\';
    %     DataPathStr='\Data\Mushrooms_All_centralization_2part_(2),(3)\';
    % DataPathStr='\Data\cervicalCellImage\';
    
    % % ------载入鸢尾花数据集------
    % load('\Data\iris-0.9.mat');
    % dataSet=iris;
    % input=dataSet(:,1:4);
    % output=dataSet(:,5:7);
    % % select=[1:50,101:150];
    % % input=input(select,:);
    % % output=output(select,:);
    
    % % ------载入红酒数据集------
    % load('\Data\wine-0.9.mat');
    % dataSet=wine;
    % input=dataSet(:,1:13);
    % output=dataSet(:,14:16);
    
    pathStr=['E:',DataPathStr];% 数据库存储路径
    
    
    %% 设置网络参数
    
    % CNN_StructPara={{参数表1};{参数表2};...;{参数表n}};
    % 参数包括：
    % 每层类型layerType(卷积层='convLayer',降采样层='subSampLayer'，全连接层='fullConnLayer'，输入层='inputLayer'，批归一化层='batchNormalizationLayer'，dropout层='dropoutLayer'),
    % 激活函数actFun(actFun='sigmoid'/'tanhFun'/'emptyFun'/'rectifier'/'softplus')
    % 特征映射数量featureMapsNum、神经元个数cellNum、卷积核尺寸(采样分辨率)size、学习率learningRate、冲量momentum
    % 与上层连接方式connType(卷积层需要此参数,全连接='full',局部连接='local')
    % 是否直接连接而不使用权值、偏置isDirConn(isDirConn=1/0)，此功能测试中，暂不开通
    % 采样方式sampType(sampType='max'/'mean')
    
    % 不同类型的层的参数表格式如下：
    % {输入层参数表}={layerType,actFun}
    % {卷积层参数表}={layerType,actFun,learningRate,momentum,size,connType,connMatrix}，若connType='local'或
    % {卷积层参数表}={layerType,actFun,learningRate,momentum,size,connType,featureMapsNum}，若connType='full'
    % {降采样层参数表}={layerType,actFun,learningRate,momentum,size,isDirConn,sampType}
    % {全连接层参数表}={layerType,actFun,learningRate,momentum,cellNum}
    % {批归一化层参数表}={layerType,actFun,learningRate,momentum,bnEpsilon}
    
    defLR=0.09;% 默认学习率
    defCovLR=defLR;% 默认卷积层学习率
    defBNLR=0.005;% 默认BN层学习率
    defMomentum=0.9;% 默认冲量
    defCovMomentum=defMomentum;% 默认卷积层冲量
    defDropoutRate=0.4;% 默认dropout比例
    defCovDropoutRate=1.2;% 默认卷积层dropout比例
    defMaxWeightRange=15;% 默认的权重最大变化范围
    defInitWeightRange=0.15;% 默认的权重初始化时的随机范围
    defInitCovWeightRange=0.15;% 默认的卷积层权重初始化时的随机范围
    
    % 卷积层与下采样层间的连接矩阵，行代表本层（卷积层）的特征映射图号，列代表每个本层特征映射图与上层的哪几个特征映射图间有连接
    connMatrix1=[
        1,1,1,0,0,0;
        0,1,1,1,0,0;
        0,0,1,1,1,0;
        0,0,0,1,1,1;
        1,0,0,0,1,1;
        1,1,0,0,0,1;
        1,1,1,1,0,0;
        0,1,1,1,1,0;
        0,0,1,1,1,1;
        1,0,0,1,1,1;
        1,1,0,0,1,1;
        1,1,1,0,0,1;
        0,1,1,0,1,1;
        1,1,0,1,1,0;
        1,0,1,1,0,1;
        1,1,1,1,1,1;
        ];
    
    % ------设置网络结构参数------
    CNN_StructPara={
        {'inputLayer'};
        %     {'dropoutLayer',defCovDropoutRate};
        {'convLayer','rectifier',defCovLR,defCovMomentum,defInitCovWeightRange,defMaxWeightRange,3,'full',6};
        %     {'dropoutLayer',defCovDropoutRate};
%         {'batchNormalizationLayer','rectifier',defBNLR,defMomentum};
        {'subSampLayer','emptyFun',defLR,defMomentum,defInitWeightRange,defMaxWeightRange,2,1,'max'};
        %         {'convLayer','emptyFun',defCovLR,defCovMomentum,defInitCovWeightRange,defMaxWeightRange,5,'local',connMatrix1};
        {'convLayer','rectifier',defCovLR,defCovMomentum,defInitCovWeightRange,defMaxWeightRange,2,'full',16};
        %     {'dropoutLayer',defCovDropoutRate};
%         {'batchNormalizationLayer','rectifier',defBNLR,defMomentum};
        {'subSampLayer','emptyFun',defLR,defMomentum,defInitWeightRange,defMaxWeightRange,2,1,'max'};
%         {'convLayer','emptyFun',defCovLR,defCovMomentum,defInitCovWeightRange,defMaxWeightRange,2,'full',30};
%         {'batchNormalizationLayer','rectifier',defBNLR,defMomentum};
%         {'subSampLayer','emptyFun',defLR,defMomentum,defInitWeightRange,defMaxWeightRange,2,1,'max'};
%         {'convLayer','emptyFun',defCovLR,defCovMomentum,defInitCovWeightRange,defMaxWeightRange,2,'full',30};
%         {'batchNormalizationLayer','rectifier',defBNLR,defMomentum};
%         {'subSampLayer','emptyFun',defLR,defMomentum,defInitWeightRange,defMaxWeightRange,2,1,'max'};
%         {'convLayer','emptyFun',defCovLR,defCovMomentum,defInitCovWeightRange,defMaxWeightRange,2,'full',40};
%         {'convLayer','emptyFun',defCovLR,defCovMomentum,defInitCovWeightRange,defMaxWeightRange,2,'full',40};
%         {'batchNormalizationLayer','rectifier',defBNLR,defMomentum};
%         {'subSampLayer','emptyFun',defLR,defMomentum,defInitWeightRange,defMaxWeightRange,2,1,'max'};
        {'fullConnLayer','rectifier',defLR,defMomentum,defInitWeightRange,defMaxWeightRange,120};
%         {'dropoutLayer',defDropoutRate};
%         {'batchNormalizationLayer','rectifier',defBNLR,defMomentum};
        {'fullConnLayer','rectifier',defLR,defMomentum,defInitWeightRange,defMaxWeightRange,84};
%         {'dropoutLayer',defDropoutRate};
%         {'batchNormalizationLayer','rectifier',defBNLR,defMomentum};
        {'fullConnLayer','sigmoid',defLR,defMomentum,defInitWeightRange,defMaxWeightRange,outputSize}
        };
    
    %% 创建网络并训练
    
    % cnn1=buildCNN(CNN_StructPara,'vector',inputSize);% 创建一维向量的网络
    % cnn1=trainCNN(cnn1,99999999,1,input,output,0.7);% 一维
    
    % cnn1=buildCNN(CNN_StructPara,'image',inputSize);% 创建二维图像的网络
    [~]=CNN_Train(CNN_StructPara,maxIterNum,costFunction,batchSize,isShowProgress,accuracy,netSavePath,pathStr,outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal,isLoadExistCNN,regularizationLamda,regularizationType);% 二维图象
    
    % % 打印网络基本信息
    % for i=1:length(cnn1)
    %     cnn1{i}
    % end
    % % pause
    
    
    
end
end