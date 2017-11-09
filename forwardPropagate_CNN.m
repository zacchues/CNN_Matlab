function [res,cnn]=forwardPropagate_CNN(cnn,input,batchNumI,batchNum,singleStepIndex)
batchSize=size(input,1);% 本批样本数量

if nargin==4
    beginI=1;
    endI=length(cnn);
elseif nargin==5
    beginI=singleStepIndex;
    endI=singleStepIndex;
end
for i=beginI:endI% 根据结构参数，对应计算输出每一层网络
    if isfield(cnn{i},'actFun')
        if strcmp(cnn{i}.actFun,'emptyFun')% 获取当前层对应激活函数
            actFun=@ emptyFun;
        elseif strcmp(cnn{i}.actFun,'sigmoid')
            actFun=@ sigmoid;
        elseif strcmp(cnn{i}.actFun,'tanhFun')
            actFun=@ tanhFun;
        elseif strcmp(cnn{i}.actFun,'rectifier')
            actFun=@ rectifier;
        elseif strcmp(cnn{i}.actFun,'softplus')
            actFun=@ softplus;
        end
    end
    
    if strcmp(cnn{i}.layerType,'inputLayer')% ------------输入层计算输出------------
        cnn{i}.output(1:batchSize,:)=input;% 输入层的输出直接等于网络接受的输入数据
        
    elseif strcmp(cnn{i}.layerType,'convLayer')% ------------卷积层计算输出------------
        for bs=1:batchSize% 依层计算批中的每个样本
            if strcmp(cnn{i}.connType,'full')% 如果为全连接式，则当前层所有导数与前层所有特征映射相连
                
                convKernel=cnn{i}.weights;% 获取本层卷积核
                for li=1:cnn{i}.featureMapNum% 对每个本层的特征映射
                    tempOutput=zeros(cnn{i}.outputFMSize);
                    for pli=1:size(cnn{i-1}.output,2)% 对每个上层的特征映射
                        % 使用matlab自带的卷积函数，会自动将卷积核进行180度旋转，所以输入时需要人工旋转一次与之相抵消
                        tempOutput=tempOutput+convn(cnn{i-1}.output{bs,pli},convKernel{pli,li}(end:-1:1,end:-1:1),'valid');
                    end
                    cnn{i}.output{bs,li}=actFun(tempOutput+cnn{i}.bias{li});
                end
            elseif strcmp(cnn{i}.connType,'local')% 如果为局部连接式，则当前层所有导数与前层所有特征映射相连
                
                connIndexMatrix=cnn{i}.connIndexMatrix;
                
                convKernel=cnn{i}.weights;% 获取本层卷积核
                
                for li=1:size(cnn{i}.output,2)% 对每个本层的特征映射
                    tempOutput=zeros(cnn{i}.outputFMSize);
                    for pli=1:size(cnn{i-1}.output,2)% 对每个上层的特征映射
                        if connIndexMatrix(li,pli)~=0
                            % 使用matlab自带的卷积函数，会自动将卷积核进行180度旋转，所以输入时需要人工旋转一次与之相抵消
                            tempOutput=tempOutput+convn(cnn{i-1}.output{bs,pli},convKernel{pli,li}(end:-1:1,end:-1:1),'valid');
                        end
                    end
                    cnn{i}.output{bs,li}=actFun(tempOutput+cnn{i}.bias{li});
                end
            end
        end
        
    elseif strcmp(cnn{i}.layerType,'subSampLayer')% ------------降采样层计算输出------------
        for bs=1:batchSize% 依层计算批中的每个样本
            featuresMapNum=size(cnn{i}.output,2);% 特征映射数量
            
            [l1,l2]=size(cnn{i}.output{1,1});% 本层特征映射尺寸
            sampSize=cnn{i}.size;% 获取采样分辨率
            perLayerOutputSize=size(cnn{i-1}.output{1,1},1);% 获取上层特征映射尺寸
            
            for fm=1:featuresMapNum% 对本层与上层的每个对应特征映射
                tempFM=cnn{i-1}.output{bs,fm};
                
                if strcmp(cnn{i}.sampType,'max')% 根据采样方式生成当前层各元素的值
                    for i1=1:l1
                        for i2=1:l2
                            % 如采样分辨率与上层特征映射尺寸不匹配，则将剩下的部分作为不完全采样区域处理，所以在反向传播时需要先判断是否能整除来寻找对应的边
                            if (i1*sampSize)<perLayerOutputSize
                                upIndex1=(i1*sampSize);
                            else
                                upIndex1=perLayerOutputSize;
                            end
                            if (i2*sampSize)<perLayerOutputSize
                                upIndex2=(i2*sampSize);
                            else
                                upIndex2=perLayerOutputSize;
                            end
                            sampRegion=tempFM(((i1-1)*sampSize+1):upIndex1,((i2-1)*sampSize+1):upIndex2);% 从上层取出采样区域保存在sampRegion中
                            cnn{i}.output{bs,fm}(i1,i2)=max(sampRegion(:));
                        end
                    end
                elseif strcmp(cnn{i}.sampType,'mean')
                    for i1=1:l1
                        for i2=1:l2
                            % 如采样分辨率与上层特征映射尺寸不匹配，则将剩下的部分作为不完全采样区域处理，所以在反向传播时需要先判断是否能整除来寻找对应的边
                            if (i1*sampSize)<perLayerOutputSize
                                upIndex1=(i1*sampSize);
                            else
                                upIndex1=perLayerOutputSize;
                            end
                            if (i2*sampSize)<perLayerOutputSize
                                upIndex2=(i2*sampSize);
                            else
                                upIndex2=perLayerOutputSize;
                            end
                            sampRegion=tempFM(((i1-1)*sampSize+1):upIndex1,((i2-1)*sampSize+1):upIndex2);% 从上层取出采样区域保存在sampRegion中
                            cnn{i}.output{bs,fm}(i1,i2)=mean(sampRegion(:));
                        end
                    end
                end
                
                if cnn{i}.isDirConn==0% 如果不为直连，则使用偏置、权值、激活函数处理输出，待完善
                end
            end
        end
    elseif strcmp(cnn{i}.layerType,'fullConnLayer')% ------------全连接层计算输出------------
        for bs=1:batchSize% 依层计算批中的每个样本
            % 光珊化上一层输出
            preLayerOutput=cnn{i-1}.output(bs,:);% 上一层输出（一维向量或矩阵集）
            
            preLayerOutput1=[];% 光珊化后结果
            preOutputSize=size(preLayerOutput{1},1);% 上层特征映射的尺寸
            
            for i1=1:length(preLayerOutput)% 光珊化，将上一层所有特征映射矩阵拉成一列并依次排列
                for i2=1:preOutputSize
                    preLayerOutput1=[preLayerOutput1,preLayerOutput{i1}(i2,:)];
                end
            end
            preLayerOutput1=preLayerOutput1';% 输出转为列向量
            
            cnn{i}.output{bs,1}=actFun([preLayerOutput1'*cnn{i}.weights{1}]'+cnn{i}.bias{1});%用激活函数、权值与偏置处理光珊化后的输入
        end
    elseif strcmp(cnn{i}.layerType,'batchNormalizationLayer')% ------------全连接层计算输出------------
        preLayerOutput=cnn{i-1}.output;%获取上一层网络输出
        beta=cnn{i}.beta;
        gamma=cnn{i}.gamma;
        epsilon=10^(-10);
        if (size(preLayerOutput{1,1},2)==1)&&(size(preLayerOutput{1,1},1)~=1)% 如果上一层是全连接层输出格式，则按全连接层的方式计算
            
            % 全连接层输出preLayerOutput为size*1的列向量，其中size为全连接层神经元数
            
            output1=nan(batchSize,size(preLayerOutput{1},1));
            
            for bs=1:batchSize% 依层计算批中的每个样本
                output1(bs,:)=preLayerOutput{bs}';% 将上层输出转为矩阵
            end
            if (batchNum~=0)% batchNum~=0说明此时是训练过程，需要计算当前批次的期望与方差
                outputExpectation=mean(output1,1);% 上层输出均值(期望)
                if (size(output1,1)==1)% 上层输出方差
                    outputVarience=ones(1,size(output1,2));% 如果批中只有一条样本，则默认方差为1
                else
                    outputVarience=var(output1);% 否则按列计算方差,返回长度为1*size的行向量
                end
            else% batchNum==0说明此时是预测过程，直接调用以前计算过所有批次的期望与方差来求平均值
                outputExpectation=cnn{i}.outputExpectationAvg{1};
                outputVarience=cnn{i}.outputVarianceAvg{1};
            end
            preLayerOutputMatrixNorm=(output1-repmat(outputExpectation,batchSize,1))./repmat((sqrt(outputVarience)+epsilon),batchSize,1);% 标准化前一层输出
            act=preLayerOutputMatrixNorm.*repmat(gamma,batchSize,1)+repmat(beta,batchSize,1);
            
            
            %             if batchNum~=0
            %             mean(act)
            %             var(act)
            %             pause
            %             end
            
            
            
            output1=actFun(act);% 对前一层输出结果进行位移并用激活函数处理
            %             output1=actFun(preLayerOutputMatrixNorm);% 对前一层输出结果进行位移并用激活函数处理
            
            
            for bs=1:batchSize% 还原为原始输出格式
                cnn{i}.preLayerOutputNorm{bs}=preLayerOutputMatrixNorm(bs,:)';
                cnn{i}.output{bs}=output1(bs,:)';
            end
            
        else% 非全连接层的前一层输出为用cell包装的特征映射矩阵，为batchSize*featureMapNum
            
            % 非全连接层输出preLayerOutput为batchSize*featureMapNum的cell，其中每个cell中的元素为一个特征映射矩阵
            
            featureMapNum=size(preLayerOutput,2);
            
            if (batchNum~=0)% batchNum~=0说明此时是训练过程，需要计算当前批次的各个特征映射矩阵的期望与方差
                % 非全连接层的上层输出期望与方差均用batchSize*featureMapNum的矩阵来存储
                outputExpectation=zeros(1,featureMapNum);
                outputVarience=zeros(1,featureMapNum);
                for bs=1:batchSize
                    for fmi=1:featureMapNum
                        outputExpectation(1,fmi)=outputExpectation(1,fmi)+mean(preLayerOutput{bs,fmi}(:));% 上层输出的特征映射矩阵中所有元素的均值(期望)
                        % 注意：此处使用var来计算的方差为方差期望值的无偏估计值，此处为图省事就直接用无偏来代替真实值了吧，反正没什么太大影响
                        outputVarience(1,fmi)=outputVarience(1,fmi)+var(preLayerOutput{bs,fmi}(:));% 上层输出的特征映射矩阵中所有元素的方差(无偏)
                    end
                    if bs==batchSize
                        outputExpectation=outputExpectation./(batchSize);
                        outputVarience=outputVarience./(batchSize);
                    end
                end
                if(sum(size(preLayerOutput{1,1}))==2)% 如果上层输出为只有一个元素的特征映射图，则默认方差为1
                    outputVarience=ones(1,featureMapNum);
                end
            else% batchNum==0说明此时是预测过程，直接调用以前计算过所有批次的期望与方差来求平均值
                outputExpectation=cnn{i}.outputExpectationAvg{1};
                outputVarience=cnn{i}.outputVarianceAvg{1};
            end
            
            
            for bs=1:batchSize
                for fmi=1:featureMapNum
                    cnn{i}.preLayerOutputNorm{bs,fmi}=(preLayerOutput{bs,fmi}-outputExpectation(1,fmi))./sqrt(outputVarience(1,fmi)+epsilon);% 标准化前一层输出
                    
                    cnn{i}.output{bs,fmi}=actFun(cnn{i}.preLayerOutputNorm{bs,fmi}*gamma(1,fmi)+beta(1,fmi));% 对前一层输出结果进行位移并用激活函数处理
                end
            end
        end
        
        if (batchNum~=0)% batchNum~=0说明此时是训练过程，需要记录当前批次的期望与方差
            
            cnn{i}.outputExpectation{batchNumI,1}=outputExpectation;% 存储上层输出均值(期望)
            cnn{i}.outputVarience{batchNumI,1}=outputVarience;% 存储上层输出方差
            
            if(batchNumI==batchNum)
                outputExpectationAvg=zeros(size(cnn{i}.outputExpectation{1,1}));% 存储上层输出期望
                outputVarienceAvg=zeros(size(cnn{i}.outputVarience{1,1}));% 存储上层输出方差
                for bni=1:batchNum% 计算所有批中期望与方差的平均值
                    outputExpectationAvg=outputExpectationAvg+cnn{i}.outputExpectation{bni,1};% 存储上层输出均值(期望)
                    outputVarienceAvg=outputVarienceAvg+cnn{i}.outputVarience{bni,1};% 存储上层输出方差
                    if(bni==batchNum)
                        outputExpectationAvg=outputExpectationAvg./batchNum;% 取全体样本期望值的均值
                        outputVarienceAvg=(outputVarienceAvg./batchNum)*(batchSize/(batchSize-1));% 取全体批样本方差的无偏估计
                    end
                end
                cnn{i}.outputExpectationAvg(1)={outputExpectationAvg};% 记录所有训练批次的期望平均值
                cnn{i}.outputVarianceAvg(1)={outputVarienceAvg};% 记录所有训练批次的方差平均值
            end
        end
    elseif strcmp(cnn{i}.layerType,'dropoutLayer')% ------------dropout层计算输出------------
        
        preLayerOutput=cnn{i-1}.output;
        if(cnn{i}.dropoutRate>0)%如果dropoutRate==0则不进行dropout
            for bs=1:batchSize% 对本batch中的每一个样本
                
                if (size(preLayerOutput{1,1},2)==1)&&(size(preLayerOutput{1,1},1)~=1)% 如果上一层是全连接层输出格式，则按全连接层的方式计算
                    if (batchNum~=0)% batchNum~=0说明此时是训练过程，需要按概率对输出进行dropout
                        cnn{i}.dropoutMask{bs,1}=random('uniform',0,1,size(preLayerOutput{1,1}))>cnn{i}.dropoutRate;% 计算当前batch中所有output的dropoutMask
                        cnn{i}.output{bs,1}=preLayerOutput{bs,1}.*cnn{i}.dropoutMask{bs,1};% 对当前样本输出做dropout
                    else% batchNum==0说明此时是预测过程，需要按比例放缩
                        cnn{i}.output{bs,1}=preLayerOutput{bs,1}*(1-cnn{i}.dropoutRate);
                    end
                    
                else% 非全连接层的前一层输出为用cell包装的特征映射矩阵，为batchSize*featureMapNum
                    
                    featureMapNum=size(preLayerOutput,2);
                    
                    for fmi=1:featureMapNum
                        if (batchNum~=0)% batchNum~=0说明此时是训练过程，需要按概率对输出进行dropout
                            %                             cnn{i}.dropoutMask{bs,fmi}=random('uniform',0,1,size(preLayerOutput{1,1}))>cnn{i}.dropoutRate;% 计算当前batch中所有output的dropoutMask


                            if (cnn{i}.dropoutRate>1)
                                if(isnan(cnn{i}.dropoutMask{bs,fmi}(1))==1)
                                    cnn{i}.dropoutMask{bs,fmi}=ones(size(preLayerOutput{1,1}))*(rand>(cnn{i}.dropoutRate-floor(cnn{i}.dropoutRate)));% 把当前特征映射图与上下两层的连接按dropoutRate概率清零
                                end
                            else
                                cnn{i}.dropoutMask{bs,fmi}=ones(size(preLayerOutput{1,1}))*(rand>cnn{i}.dropoutRate);% 把当前特征映射图与上下两层的连接按dropoutRate概率清零
                            end
%                             cnn{i}.dropoutMask{bs,fmi}
                            
                            cnn{i}.output{bs,fmi}=preLayerOutput{bs,fmi}.*cnn{i}.dropoutMask{bs,fmi};% 对当前样本输出做dropout
                            
                        else% batchNum==0说明此时是预测过程，需要按比例放缩
                            cnn{i}.output{bs,fmi}=preLayerOutput{bs,fmi}.*(1-(cnn{i}.dropoutRate-floor(cnn{i}.dropoutRate)));
                        end
                    end
                end
                
            end
        else% 如果不进行dropout则输入等于上层输出
            cnn{i}.output=preLayerOutput;
        end
        
        
    end
end

res=cnn{end}.output;
end