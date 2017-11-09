function cnn=buildCNN(cnnStructure,inputType,inputSize,imgChNum,batchSize,batchNum)

cnn=cell(size(cnnStructure,1),1);
for i=1:length(cnnStructure)% 根据结构参数，对应初始化每一层网络
    
    if strcmp(cnnStructure{i}{1},'inputLayer')% ------------输入层初始化------------
        tempLayer.layerType=cnnStructure{i}{1};% 初始化该层类型
%         tempLayer.err={};% 输出端原始误差（不是残差，不可直接用来调整权值，用来通过反向传播计算残差）
%         tempLayer.learningRate=0;% 该层学习率
        tempLayer.weights={};
        tempLayer.bias={};
        tempLayer.outputFMSize=inputSize;% 本层输出的特征映射图的尺寸
        
        if strcmp(inputType,'image')% 如果输入类型是图片
            for bs=1:batchSize% batch中对每个批样本batch sample分别初始化
                for fmn=1:imgChNum% 初始化卷积层输出、偏置值、残差，每个特征映射用一个矩阵表示
                    tempLayer.output{bs,fmn}=nan(tempLayer.outputFMSize);% 初始化输入图形的存储空间
                end
            end
        elseif strcmp(inputType,'vector')% 如果输入类型是一维向量
            for bs=1:batchSize% batch中对每个批样本batch sample分别初始化
                tempLayer.output{bs}=nan(tempLayer.outputFMSize,1);% 初始化一维向量的存储空间
            end
        end
        
        cnn(i)={tempLayer};% 添加一层
        
    elseif strcmp(cnnStructure{i}{1},'convLayer')% ------------卷积层初始化------------
        tempLayer.layerType=cnnStructure{i}{1};% 该层类型
        tempLayer.actFun=cnnStructure{i}{2};% 该层激活函数
        tempLayer.learningRate=cnnStructure{i}{3};% 该层学习率
        tempLayer.momentum=cnnStructure{i}{4};% 该层冲量
        initWeightRange=cnnStructure{i}{5};% 该层权重初始化时的随机范围
        tempLayer.maxWeightRange=cnnStructure{i}{6};% 该层权重的最大可变范围
        tempLayer.kernelSize=cnnStructure{i}{7};% 该层卷积核尺寸
        tempLayer.connType=cnnStructure{i}{8};% 该层连接类型
        
        tempLayer.outputFMSize=size(cnn{i-1}.output{1,1},1)-tempLayer.kernelSize+1;% 通过上层输出矩阵尺寸计算本层特征映射尺寸
        
        if strcmp(tempLayer.connType,'full')% 根据连接方式确定需要的特征映射数量
            tempLayer.featureMapNum=cnnStructure{i}{9};% 该卷积层特征映射数量
            for fmn=1:tempLayer.featureMapNum% 初始化卷积层输出、偏置值、残差，每个特征映射用一个矩阵表示
                tempLayer.bias{fmn}=zeros(tempLayer.outputFMSize);
                tempLayer.deltaBias{fmn}=zeros(tempLayer.outputFMSize);% 初始化阈值改变量
                for bs=1:batchSize% batch中对每个批样本batch sample分别初始化
                    tempLayer.output{bs,fmn}=nan(tempLayer.outputFMSize);
                    tempLayer.err{bs,fmn}=nan(tempLayer.outputFMSize);
                end
            end
            
            preLayerOutputNum=size(cnn{i-1}.output,2);% 获取上层输出的特征映射图数量
            
            tempLayer.weights=cell(preLayerOutputNum,tempLayer.featureMapNum);% 根据前一层输出数量、本层特征映射数量、卷积核尺寸来初始化本层权值
            % cell中行号为上一层特征映射号，列号为本层特征映射号
            for i1=1:preLayerOutputNum
                for i2=1:tempLayer.featureMapNum
                    % weights的行号代表上层对应的特征映射层，列号代表本层对应的特征映射层，可以这两个编号来进行检索
                    tempLayer.weights{i1,i2}=random('uniform',-initWeightRange,initWeightRange,tempLayer.kernelSize,tempLayer.kernelSize);% 随机初始化权值
                    tempLayer.deltaWeights{i1,i2}=zeros(tempLayer.kernelSize,tempLayer.kernelSize);% 初始化权值改变量
                end
            end
        elseif strcmp(tempLayer.connType,'local')% 根据连接矩阵生成特征映射与连接的卷积核等
            tempLayer.connIndexMatrix=cnnStructure{i}{9};% 获取连接矩阵
            preLayerOutputNum=size(cnn{i-1}.output,2);
            if size(tempLayer.connIndexMatrix,2)~=preLayerOutputNum% 检验输入的连接矩阵的前层连接数量是否与前一层特征映射图数量相同，如不同则输出警告信息
                inf1='output size cannot match,please check your input Matrix.';
                inf2=' Current layer:';
                preLayerFeatureMapNum=' PreLayer feature map num:';
                currentLayerFeatureMapNum=' Current feature map num:';
                disp([inf1,inf2,num2str(i),preLayerFeatureMapNum,num2str(size(tempLayer.connIndexMatrix,2)),currentLayerFeatureMapNum,num2str(length(cnn{i-1}.output))]);
                pause;
            end
            
            tempLayer.featureMapNum=size(tempLayer.connIndexMatrix,1);% 连接矩阵中的列表示前一层特征映射图，行表示本层特征映射图
            
            for fmn=1:tempLayer.featureMapNum% 初始化卷积层输出、偏置值、残差，每个特征映射用一个矩阵表示
                tempLayer.bias{fmn}=zeros(tempLayer.outputFMSize);
                tempLayer.deltaBias{fmn}=zeros(tempLayer.outputFMSize);% 初始化阈值改变量
                for bs=1:batchSize% batch中对每个批样本batch sample分别初始化
                    tempLayer.output{bs,fmn}=nan(tempLayer.outputFMSize);
                    tempLayer.err{bs,fmn}=nan(tempLayer.outputFMSize);
                end
            end
            
            tempLayer.weights=cell(preLayerOutputNum,tempLayer.featureMapNum);% 根据前一层输出数量、本层特征映射数量、卷积核尺寸来初始化本层权值
            
            for i1=1:preLayerOutputNum% cell中行号为上一层特征映射号，列号为本层特征映射号
                for i2=1:tempLayer.featureMapNum
                    % weights的行号代表上层对应的特征映射层，列号代表本层对应的特征映射层，可以这两个编号来进行检索
                    if (tempLayer.connIndexMatrix(i2,i1))~=0
                        tempLayer.weights{i1,i2}=random('uniform',-initWeightRange,initWeightRange,tempLayer.kernelSize,tempLayer.kernelSize);% 随机初始化权值
                        tempLayer.deltaWeights{i1,i2}=zeros(tempLayer.kernelSize,tempLayer.kernelSize);% 初始化权值改变量
                    else
                        tempLayer.weights{i1,i2}=nan;
                    end
                end
            end
        end
        cnn(i)={tempLayer};
        
    elseif strcmp(cnnStructure{i}{1},'subSampLayer')% ------------降采样层初始化------------
        tempLayer.layerType=cnnStructure{i}{1};% 初始化该层类型
        tempLayer.actFun=cnnStructure{i}{2};
        tempLayer.learningRate=cnnStructure{i}{3};
        tempLayer.momentum=cnnStructure{i}{4};
        initWeightRange=cnnStructure{i}{5};% 该层权重初始化时的随机范围
        tempLayer.maxWeightRange=cnnStructure{i}{6};% 该层权重的最大可变范围
        tempLayer.size=cnnStructure{i}{7};
        tempLayer.isDirConn=cnnStructure{i}{8};
        tempLayer.sampType=cnnStructure{i}{9};
        tempLayer.outputFMSize=ceil(size(cnn{i-1}.output{1,1},1)/tempLayer.size);% 通过上层输出矩阵尺寸与采样分辨率计算本层特征映射矩阵尺寸
        
        
        
        for fmn=1:size(cnn{i-1}.output,2)% 初始化降采样层输出、偏置值、残差
            tempLayer.bias{fmn}=zeros(tempLayer.outputFMSize);
            for bs=1:batchSize% batch中对每个批样本batch sample分别初始化
                tempLayer.output{bs,fmn}=nan(tempLayer.outputFMSize);
                tempLayer.err{bs,fmn}=nan(tempLayer.outputFMSize);
            end
        end
        tempLayer.weights={};% 直连不需要设置权值，或默认全为1
        tempLayer.deltaWeights=0;
        tempLayer.deltaBias=0;
        
        cnn(i)={tempLayer};
        
    elseif strcmp(cnnStructure{i}{1},'fullConnLayer')% ------------全连接层初始化------------
        tempLayer.layerType=cnnStructure{i}{1};% 初始化该层类型
        tempLayer.actFun=cnnStructure{i}{2};
        tempLayer.learningRate=cnnStructure{i}{3};
        tempLayer.momentum=cnnStructure{i}{4};
        initWeightRange=cnnStructure{i}{5};% 该层权重初始化时的随机范围
        tempLayer.maxWeightRange=cnnStructure{i}{6};% 该层权重的最大可变范围
        outputSize=cnnStructure{i}{7};
        
        tempLayer.bias{1}=zeros(outputSize,1);
        tempLayer.deltaBias{1}=zeros(outputSize,1);% 初始化阈值改变量
        for bs=1:batchSize% batch中对每个批样本batch sample分别初始化
            tempLayer.bias{bs}=zeros(outputSize,1);
            tempLayer.deltaBias{bs}=zeros(outputSize,1);% 初始化阈值改变量
            tempLayer.output{bs,1}=nan(outputSize,1);% 初始化全连接层输出、偏置值、残差
            tempLayer.err{bs,1}=nan(outputSize,1);
        end
        
        % 光珊化上一层输出
        preLayerOutput=cnn{i-1}.output;% 上一层输出（一维列向量或矩阵集）
        
        %         preLayerOutput1={};% 光珊化后结果
        %         preOutputSize=cnn{i-1}.outputFMSize;% 上层特征映射的尺寸
        preLayerFeatureMapNum=size(preLayerOutput,2);% 上层特征映射数量，如果只是一行列向量(全连接层)则此值为1
        if max(size(preLayerOutput))~=1||min(size(preLayerOutput{1}))~=1% 如果输入为多个特征映射层
            %
            %             for bs=1:batchSize% batch中对每个批样本batch sample分别初始化
            %                 tempOutput=[];
            %                 for i1=1:preLayerFeatureMapNum% 光珊化，将上一层所有特征映射矩阵拉成一列并依次排列
            %                     for i2=1:preOutputSize
            %                         tempOutput=[tempOutput,preLayerOutput{bs,i1}(i2,:)];
            %                     end
            %                 end
            %                 preLayerOutput1{bs}=tempOutput;
            %             end
            preLayerOutputLength=size(preLayerOutput{bs,1},1)*size(preLayerOutput{bs,1},2)*preLayerFeatureMapNum;
        else% 如果输入为单个列向量
            %             for bs=1:batchSize% batch中对每个批样本batch sample分别初始化
            %                 preLayerOutput1{bs}=preLayerOutput{bs,1};
            %             end
            preLayerOutputLength=size(preLayerOutput{1},1);
        end
        tempLayer.weights={random('uniform',-initWeightRange,initWeightRange,preLayerOutputLength,outputSize)};% 每行代表上层的一个结点对应本层所有结点的权值
        tempLayer.deltaWeights={zeros(preLayerOutputLength,outputSize)};% 初始化权值改变量
        cnn(i)={tempLayer};
        
    elseif strcmp(cnnStructure{i}{1},'batchNormalizationLayer')% ------------batch normalization层初始化------------
        tempLayer.layerType=cnnStructure{i}{1};% 初始化该层类型
        tempLayer.actFun=cnnStructure{i}{2};
        tempLayer.output=cnn{i-1}.output;% 获取上层output
        tempLayer.err=cnn{i-1}.output;% 获取上层output
        tempLayer.outputExpectation=cell(batchNum,1);% 记录每一批的输出期望
        tempLayer.outputVariance=cell(batchNum,1);% 记录每一批的输出方差
        tempLayer.preLayerOutputNorm=cnn{i-1}.output;% 存储前一层输出的标准化变量
        
        if (size(tempLayer.output{1,1},2)==1)&&(size(tempLayer.output{1,1},1)~=1)% 根据output判断前一层是不是全连接层
%             tempLayer.beta=normrnd(0,1,1,size(tempLayer.output{1,1},1));% 以0为中心的正态分布初始化
%             tempLayer.gamma=normrnd(1,1,1,size(tempLayer.output{1,1},1));% 以1为中心的正态分布初始化
            tempLayer.beta=zeros(1,size(tempLayer.output{1,1},1));% 全0初始化
            tempLayer.gamma=ones(1,size(tempLayer.output{1,1},1));% 全1初始化
            %             tempLayer.beta=zeros(1,size(tempLayer.output{1,1},1));
            %             tempLayer.gamma=ones(1,size(tempLayer.output{1,1},1));
            tempLayer.outputExpectationAvg=cell(1,1);% 记录所有训练批次的期望平均值
            tempLayer.outputVarianceAvg=cell(1,1);% 记录所有训练批次的方差平均值
        else% 如果不是全连接层
%             tempLayer.beta=normrnd(0,1,1,size(tempLayer.output,2));% 以0为中心的正态分布初始化
%             tempLayer.gamma=normrnd(1,1,1,size(tempLayer.output,2));% 以1为中心的正态分布初始化
            tempLayer.beta=zeros(1,size(tempLayer.output,2));% 全0初始化
            tempLayer.gamma=ones(1,size(tempLayer.output,2));% 全1初始化
            tempLayer.outputExpectationAvg=cell(1,size(tempLayer.output,2));% 记录所有训练批次的期望平均值
            tempLayer.outputVarianceAvg=cell(1,size(tempLayer.output,2));% 记录所有训练批次的方差平均值
        end
        tempLayer.learningRate=cnnStructure{i}{3};
        tempLayer.momentum=cnnStructure{i}{4};
        
        cnn(i)={tempLayer};
        
    elseif strcmp(cnnStructure{i}{1},'dropoutLayer')% ------------dropout层初始化------------
        
        tempLayer.layerType=cnnStructure{i}{1};% 初始化该层类型
        tempLayer.dropoutRate=cnnStructure{i}{2};
        tempLayer.output=cnn{i-1}.output;% 获取上层output来初始化本层output
        tempLayer.err=cnn{i-1}.output;% 获取上层err来初始化本层output
        tempLayer.dropoutMask=cnn{i-1}.output;% 初始化dropoutMask，大小与输出及误差矩阵相同，用来标记哪个结点输出应该变为0
        
        cnn(i)={tempLayer};
    end
    
    clear tempLayer;
end

end