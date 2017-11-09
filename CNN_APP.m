function CNN_APP
% ���������ѵ��������

mainPath = '.';% !!!���г���ǰ�������ú���Ŀ¼·��
addpath(genpath(mainPath));

% �趨��ǰ����������Ԥ�⻹������ѵ��
% CNN_Type = 'test';% Ԥ��
CNN_Type = 'train';% ѵ��

% �趨����洢λ��
netSavePath = [mainPath , '/cnn_Save/Mnist_cnn.mat'];% ����洢λ��

% Ԥ�������pre-process
imgSize = 32;% ͼ���һ���ߴ�
isDoGray = 0;% �Ƿ����ҶȻ�
isDoBW = 0;% �Ƿ�����ֵ��
isDoColorReversal = 0;% �Ƿ�����ɫ��ת

if strcmp(CNN_Type , 'test')% Ԥ��
    CNN_Test(netSavePath , imgSize , isDoGray , isDoBW , isDoColorReversal);

elseif strcmp(CNN_Type , 'train') % ѵ��
    %% [1]���û���ѵ������training parameters
    
    batchSize = 10;% ѵ������һ��batch�а�������������
    
    % ѡ����ۺ���
    costFunction = @ MSE;
    % costFunction = @ softmax_Loss;
    
    % �������򻯷�ʽ�����������򻯣�'nan'��L1���򻯣�'L1',L2���򻯣�'L2'
    % ���ַ�ʽĬ�϶�����Ȩֵ��С
    regularizationType = 'L1';
    regularizationLamda = 0.1;% ���򻯲���
    
    maxIterNum = 10000;% ѵ����������
    accuracy = 2;% ������ʾ���ȣ�������λС����λ����
    isShowProgress = 1;% �Ƿ���ʾʵʱ��������������ߣ�1Ϊ��ʾ��0Ϊ����ʾ
    isLoadExistCNN = 0;% �Ƿ�������������,0��ʾ�½����磨�����룩
    
    
    % �������ά�ȣ�����������
    outputSize = 10;
    
    % ------����ͼ�����ݿ�------
    
    % ͼ������·��
    % DataPathStr = '/Data/Mnist/';
    DataPathStr = '/Data/70/';
    
    pathStr = [mainPath , DataPathStr];% ���ݿ�洢·��
    
    %% [2]���������������������ṹ
    
    % CNN_StructPara = {{������1};{������2};...;{������n}};
    % ����������
    % ÿ������layerType(�����='convLayer',��������='subSampLayer'��ȫ���Ӳ�='fullConnLayer'�������='inputLayer'������һ����='batchNormalizationLayer'��dropout��='dropoutLayer'),
    % �����actFun(actFun='sigmoid'/'tanhFun'/'emptyFun'/'rectifier'/'softplus')
    % ����ӳ������featureMapsNum����Ԫ����cellNum������˳ߴ�(�����ֱ���)size��ѧϰ��learningRate������momentum
    % ���ϲ����ӷ�ʽconnType(�������Ҫ�˲���,ȫ����='full',�ֲ�����='local')
    % �Ƿ�ֱ�����Ӷ���ʹ��Ȩֵ��ƫ��isDirConn(isDirConn=1/0)���˹��ܲ����У��ݲ���ͨ
    % ������ʽsampType(sampType='max'/'mean')
    
    % ��ͬ���͵Ĳ�Ĳ������ʽ���£�
    % {����������}={layerType,actFun}
    % {����������}={layerType,actFun,learningRate,momentum,size,connType,connMatrix}����connType='local'��
    % {����������}={layerType,actFun,learningRate,momentum,size,connType,featureMapsNum}����connType='full'
    % {�������������}={layerType,actFun,learningRate,momentum,size,isDirConn,sampType}
    % {ȫ���Ӳ������}={layerType,actFun,learningRate,momentum,cellNum}
    % {����һ���������}={layerType,actFun,learningRate,momentum,bnEpsilon}
    
    defLR = 0.09;% Ĭ��ѧϰ��
    defCovLR = defLR;% Ĭ�Ͼ����ѧϰ��
    defBNLR = 0.005;% Ĭ��BN��ѧϰ��
    defMomentum = 0.9;% Ĭ�ϳ���
    defCovMomentum = defMomentum;% Ĭ�Ͼ�������
    defDropoutRate = 0.4;% Ĭ��dropout����
    defCovDropoutRate = 1.2;% Ĭ�Ͼ����dropout����
    defMaxWeightRange = 15;% Ĭ�ϵ�Ȩ�����仯��Χ
    defInitWeightRange = 0.15;% Ĭ�ϵ�Ȩ�س�ʼ��ʱ�������Χ
    defInitCovWeightRange = 0.15;% Ĭ�ϵľ����Ȩ�س�ʼ��ʱ�������Χ
    
    % ��������²����������Ӿ����д����㣨����㣩������ӳ��ͼ�ţ��д���ÿ����������ӳ��ͼ���ϲ���ļ�������ӳ��ͼ��������
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
    
    % ------��������ṹ����------
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
    
    %% [3]�������粢ѵ��
    
    [~] = CNN_Train(CNN_StructPara,maxIterNum,costFunction,batchSize,isShowProgress,accuracy,netSavePath,pathStr,outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal,isLoadExistCNN,regularizationLamda,regularizationType);% ��άͼ��
    
    %% [4]��ӡ���������Ϣ
    % for i = 1 : length(cnn1)
    %     cnn1{i}
    % end
    % % pause
        
end
end