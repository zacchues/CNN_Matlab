function CNN_APP
% 卷积神经网络训练主程序

mainPath = '.';% !!!运行程序前请先设置好主目录路径
addpath(genpath(mainPath));

% 设定当前网络是用来预测还是用来训练
% CNN_Type = 'test';% 预测
CNN_Type = 'train';% 训练

% 设定网络存储位置
netSavePath = [mainPath , '/cnn_Save/Mnist_cnn.mat'];% 网络存储位置

% 预处理参数pre-process
imgSize = 32;% 图像归一化尺寸
isDoGray = 0;% 是否做灰度化
isDoBW = 0;% 是否做二值化
isDoColorReversal = 0;% 是否做颜色反转

if strcmp(CNN_Type , 'test')% 预测
    CNN_Test(netSavePath , imgSize , isDoGray , isDoBW , isDoColorReversal);

elseif strcmp(CNN_Type , 'train') % 训练
    %% [1]设置基本训练参数training parameters
    
    batchSize = 10;% 训练集的一批batch中包含多少条样本
    
    % 选择代价函数
    costFunction = @ MSE;
    % costFunction = @ softmax_Loss;
    
    % 设置正则化方式。不采用正则化：'nan'，L1正则化：'L1',L2正则化：'L2'
    % 三种方式默认都限制权值大小
    regularizationType = 'L1';
    regularizationLamda = 0.1;% 正则化参数
    
    maxIterNum = 10000;% 训练迭代次数
    accuracy = 2;% 进度显示精度（保留几位小数点位数）
    isShowProgress = 1;% 是否显示实时进度条和误差曲线，1为显示，0为不显示
    isLoadExistCNN = 0;% 是否载入现有网络,0表示新建网络（不载入）
    
    
    % 设置输出维度（分类数量）
    outputSize = 10;
    
    % ------载入图像数据库------
    
    % 图像数据路径
    % DataPathStr = '/Data/Mnist/';
    DataPathStr = '/Data/70/';
    
    pathStr = [mainPath , DataPathStr];% 数据库存储路径
    
    %% [2]设置网络参数，定义网络结构
    
    % CNN_StructPara = {{参数表1};{参数表2};...;{参数表n}};
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
    
    defLR = 0.09;% 默认学习率
    defCovLR = defLR;% 默认卷积层学习率
    defBNLR = 0.005;% 默认BN层学习率
    defMomentum = 0.9;% 默认冲量
    defCovMomentum = defMomentum;% 默认卷积层冲量
    defDropoutRate = 0.4;% 默认dropout比例
    defCovDropoutRate = 1.2;% 默认卷积层dropout比例
    defMaxWeightRange = 15;% 默认的权重最大变化范围
    defInitWeightRange = 0.15;% 默认的权重初始化时的随机范围
    defInitCovWeightRange = 0.15;% 默认的卷积层权重初始化时的随机范围
    
    % 卷积层与下采样层间的连接矩阵，行代表本层（卷积层）的特征映射图号，列代表每个本层特征映射图与上层的哪几个特征映射图间有连接
    connMatrix1 = [
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
    CNN_StructPara = {
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
    
    %% [3]创建网络并训练
    
    [~] = CNN_Train(CNN_StructPara,maxIterNum,costFunction,batchSize,isShowProgress,accuracy,netSavePath,pathStr,outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal,isLoadExistCNN,regularizationLamda,regularizationType);% 二维图象
    
    %% [4]打印网络基本信息
    % for i = 1 : length(cnn1)
    %     cnn1{i}
    % end
    % % pause
        
end
end