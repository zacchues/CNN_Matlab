function cnn=backwardPropagate_CNN(cnn,err_Output,batchNumI,epoch,trainSetNum,regularizationLamda,regularizationType)
batchSize=size(err_Output,1);% 本批样本数量
cnn{end}.err=err_Output;% 网络输出端原始误差初始化

for i=length(cnn):-1:2% 根据结构参数，对应计算输出每一层网络
    
    if isfield(cnn{i},'actFun')
        if strcmp(cnn{i}.actFun,'emptyFun')% 获取当前层对应激活函数对应的反函数
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
    
    if isfield(cnn{i},'momentum')
        momentum=cnn{i}.momentum;% 获取本层冲量
    end
    
    if isfield(cnn{i},'learningRate')
        learningRate=cnn{i}.learningRate;% 获取本层学习率
    end
    
    decayEpoch=30;
    %     decayRate=0.5;
    decayRate=1;
    if(epoch<decayEpoch&&(~strcmp(cnn{i}.layerType,'batchNormalizationLayer')))% 非BN层的学习率随epoch逐次降低
        learningRate=learningRate*(decayRate^epoch);
    else
        learningRate=learningRate*(decayRate^decayEpoch);
    end
    
    regularWeights=nan;
    
    
    if strcmp(cnn{i}.layerType,'fullConnLayer')% ------------全连接层计算------------
        residual=cell(batchSize,1);
        for bs=1:batchSize% 依层计算批中的每个样本
            % ------计算上层误差------
            weights=cnn{i}.weights{1};% 获取权值，为一向量
            err=cnn{i}.err{bs,1};% 获取当前层原始误差，为一向量
            residual{bs,1}=err.*actFun(cnn{i}.output{bs,1},'derivative');%计算本层残差：将网络输出端原始误差反向传播为输入端误差
            
            if isfield(cnn{i-1},'err')==1% 只有非输入层才需要计算上层误差
                perLayerErr=residual{bs}'*weights';% 计算前一层输出端原始误差，为一向量
                perLayerErr=perLayerErr';% 将行向量转为列向量
                size1=length(perLayerErr);% 前一层误差向量长度
                size2=size(cnn{i-1}.err,2);% 前一层误差矩阵的个数
                size3=size1/size2;% 计算每个矩阵需要截取的误差向量长度为多少
                
                if (size(cnn{i-1}.output{1,1},2)==1)&&(size(cnn{i-1}.output{1,1},1)~=1)% 如果上一层是全连接层输出格式
                    cnn{i-1}.err{bs,1}=perLayerErr;
                else
                    for ei=1:size2
                        tempErrMap=perLayerErr((ei-1)*size3+1:ei*size3);% 截取第ei个误差矩阵对应的误差向量
                        tempErrMap=reshape(tempErrMap,sqrt(size3),sqrt(size3));% 向量转化为矩阵，按列的顺序排列
                        cnn{i-1}.err{bs,ei}=tempErrMap';% 转换为按行排列并存储于上层误差中
                    end
                end
            end
            % ------调整本层权值与偏置------
            regularWeights=cnn{i}.weights;
            % 光珊化上一层输出
            preLayerOutput=cnn{i-1}.output(bs,:);% 上一层输出（一维向量或矩阵集）
            preLayerOutput1{bs}=[];% 光珊化后结果
            preOutputSize=length(preLayerOutput{1,1});% 上层特征映射的尺寸
            for i1=1:length(preLayerOutput)% 光珊化，将上一层所有特征映射矩阵拉成一列并依次排列
                for i2=1:preOutputSize
                    preLayerOutput1{bs}=[preLayerOutput1{bs},preLayerOutput{i1}(i2,:)];
                end
            end
            
            if bs==batchSize% 在计算完每批所有项后再通过计算平均值来调整各项参数
                
                
                
                renewBias=0;
                if i<length(cnn)% 判断需要不需要更新bias。如果下一层不是BN层则更新，记renewBias=1;否则为0
                    if strcmp(cnn{i+1}.layerType,'batchNormalizationLayer')==0
                        renewBias=1;
                    end
                elseif i==length(cnn)
                    renewBias=1;
                end
                
                
                
                preDeltaWeights=cnn{i}.deltaWeights{1};% 获取前一次权值改变量
                
                deltaWeightsAvg=zeros(size(residual{1},1),size(preLayerOutput1{1},2));
                residualAvg=zeros(size(residual{1}));
                
                for bs1=1:batchSize% 求各输出残差矩阵和权值改变矩阵的平均值
                    deltaWeightsAvg=deltaWeightsAvg+residual{bs1}*preLayerOutput1{bs1};% 计算权值改变量平均值
                    residualAvg=residualAvg+residual{bs1};% 计算残差的平均值
                end
                deltaWeightsAvg=deltaWeightsAvg./batchSize;
                residualAvg=residualAvg./batchSize;
                
                cnn{i}.deltaWeights{1}=learningRate.*(deltaWeightsAvg)';% 更新权值改变量
                cnn{i}.deltaBias{1}=learningRate.*residualAvg;% 更新阈值改变量
                cnn{i}.weights{1}=cnn{i}.weights{1}-(cnn{i}.deltaWeights{1}+momentum.*preDeltaWeights);% 调整权值
                
                if renewBias==1
                    preDeltaBias=cnn{i}.deltaBias{1};% 获取前一次阈值改变量
                    cnn{i}.bias{1}=cnn{i}.bias{1}-(cnn{i}.deltaBias{1}+momentum.*preDeltaBias);% 调整偏置
                end
            end
        end
    elseif strcmp(cnn{i}.layerType,'subSampLayer')% ------------降采样层计算------------
        
        for bs=1:batchSize% 依层计算批中的每个样本
            
            if isfield(cnn{i-1},'err')==1% 只有非输入层才需要计算上层误差
                % ------计算上层误差------
                
                preLayerErrSize=length(cnn{i-1}.err{bs,1});% 上层误差矩阵尺寸
                featuresMapNum=size(cnn{i}.output,2);% 特征映射或误差矩阵数量
                
                for fmi=1:featuresMapNum
                    expandedErr{fmi}=kron(cnn{i}.err{bs,fmi},ones(cnn{i}.size));% 拓展后的本层输出端原始误差矩阵，尺寸大于或等于前层误差矩阵
                end
                
                if cnn{i}.isDirConn==1% 如果为直连，则残差等于输出端原始误差，不需要通过激活函数的反函数结合输出进行计算
                    for fmi=1:featuresMapNum% 计算上层每个误差矩阵
                        cnn{i-1}.err{bs,fmi}=expandedErr{fmi}(1:preLayerErrSize,1:preLayerErrSize);
                    end
                else% 如果为非直连，则需要对上层输出进行反向传播处理，待完善
                    
                end
                
            end
            % ------调整本层权值与偏置------
            
            
            % 非直连才需要调整，此处略
        end
    elseif strcmp(cnn{i}.layerType,'convLayer')% ------------卷积层计算------------
        featuresMapNum=size(cnn{i}.output,2);% 特征映射或误差矩阵数量
        residual=cell(batchSize,featuresMapNum);
        for bs=1:batchSize% 依层计算批中的每个样本
            % ------计算上层误差------
            if strcmp(cnn{i}.connType,'local')
                connIndexMatrix=cnn{i}.connIndexMatrix;
            end
            featuresMapNum=size(cnn{i}.err,2);% 本层误差矩阵数量
            preLayerFeaturesMapNum=size(cnn{i-1}.output,2);% 上层特征映射或误差矩阵数量
            preLayerErrSize=size(cnn{i-1}.output{bs,1},1);% 上层特征映射或误差矩阵尺寸
            preLayerOutput{bs}=cnn{i-1}.output(bs,:);% 上一层输出（二矩阵集）
            for fmi=1:featuresMapNum% 计算本层输入端残差
                residual(bs,fmi)={cnn{i}.err{bs,fmi}.*actFun(cnn{i}.output{bs,fmi},'derivative')};
            end
            if isfield(cnn{i-1},'err')==1% 只有非输入层才需要计算上层误差
                for plfmi=1:preLayerFeaturesMapNum% 将每个上层特征映射矩阵对应的所有本层误差矩阵过一次
                    sumErr=zeros(preLayerErrSize);
                    for fmi=1:featuresMapNum% 累加每个上层误差矩阵中每个结点的误差
                        if strcmp(cnn{i}.connType,'local')
                            if connIndexMatrix(fmi,plfmi)~=0
                                sumErr=sumErr+conv2(residual{bs,fmi},cnn{i}.weights{plfmi,fmi},'full');
                            end
                        elseif strcmp(cnn{i}.connType,'full')
                            sumErr=sumErr+conv2(residual{bs,fmi},cnn{i}.weights{plfmi,fmi},'full');
                        end
                    end
                    cnn{i-1}.err{bs,plfmi}=sumErr;% 计算上层每个输出端原始误差矩阵的值
                end
            end
            
            % ------调整本层权值与偏置------
            regularWeights=cnn{i}.weights;
            if bs==batchSize% 在计算完每批所有项后再通过计算平均值来调整各项参数
                
                
                
                renewBias=0;
                if i<length(cnn)% 判断需要不需要更新bias。如果下一层不是BN层则更新，记renewBias=1;否则为0
                    if strcmp(cnn{i+1}.layerType,'batchNormalizationLayer')==0
                        renewBias=1;
                    end
                elseif i==length(cnn)
                    renewBias=1;
                end
                
                
                
                if renewBias==1
                    deltaBiasAvg=cell(1,featuresMapNum);% 残差矩阵平均值
                    for fmi=1:featuresMapNum% 逐个特征映射矩阵计算
                        deltaBiasAvg{fmi}=zeros(size(residual{1,1}));
                        for bs1=1:batchSize% 累加本批中各输出残差矩阵的值，并求平均值
                            deltaBiasAvg{fmi}=deltaBiasAvg{fmi}+(learningRate).*residual{bs1,fmi};% 累加批中样本的阈值改变量
                            if bs1==batchSize
                                deltaBiasAvg{fmi}=deltaBiasAvg{fmi}./batchSize;% 求每个偏置矩阵的改变量平均值
                            end
                        end
                    end
                    preDeltaBias=cnn{i}.deltaBias;% 获取前一次阈值改变量
                    cnn{i}.deltaBias=deltaBiasAvg;% 用本批平均残差更新当前的残差更新量
                end
                
                
                deltaWeightsAvg=cell(preLayerFeaturesMapNum,featuresMapNum);% 权值矩阵改变量平均值
                
                for plfmi=1:preLayerFeaturesMapNum% 将每个上层特征映射矩阵对应的所有本层误差矩阵过一次
                    for fmi=1:featuresMapNum% 逐个特征映射矩阵计算
                        deltaWeightsAvg{plfmi,fmi}=zeros(size(cnn{i}.weights{1,1}));
                        for bs1=1:batchSize% 累加本批中各样本对应的权值改变矩阵的改变量，并求平均值
                            if strcmp(cnn{i}.connType,'local')
                                if connIndexMatrix(fmi,plfmi)~=0
                                    deltaWeightsAvg{plfmi,fmi}=deltaWeightsAvg{plfmi,fmi}+(learningRate).*conv2(preLayerOutput{bs1}{plfmi},residual{bs1,fmi}(end:-1:1,end:-1:1),'valid');% 累加批中样本的权值改变量
                                end
                            elseif strcmp(cnn{i}.connType,'full')
                                deltaWeightsAvg{plfmi,fmi}=deltaWeightsAvg{plfmi,fmi}+(learningRate).*conv2(preLayerOutput{bs1}{plfmi},residual{bs1,fmi}(end:-1:1,end:-1:1),'valid');% 累加批中样本的权值改变量
                            end
                            if bs1==batchSize
                                deltaWeightsAvg{plfmi,fmi}=deltaWeightsAvg{plfmi,fmi}./batchSize;% 求每个权值矩阵的改变量平均值
                            end
                        end
                    end
                end
                
                preDeltaWeights=cnn{i}.deltaWeights;
                cnn{i}.deltaWeights=deltaWeightsAvg;
                
                for fmi=1:featuresMapNum% 调整卷积核的权值和偏置值
                    
                    if renewBias==1
                        cnn{i}.bias{fmi}=cnn{i}.bias{fmi}+cnn{i}.deltaBias{fmi}+momentum.*preDeltaBias{fmi};% 逐个调整特征映射对应的偏置
                    end
                    for plfmi=1:preLayerFeaturesMapNum% 将每个上层特征映射矩阵对应的所有本层误差矩阵过一次
                        if strcmp(cnn{i}.connType,'local')
                            if connIndexMatrix(fmi,plfmi)~=0
                                cnn{i}.weights{plfmi,fmi}=cnn{i}.weights{plfmi,fmi}-(cnn{i}.deltaWeights{plfmi,fmi}+momentum.*preDeltaWeights{plfmi,fmi});% 调整权值
                            end
                        elseif strcmp(cnn{i}.connType,'full')
                            cnn{i}.weights{plfmi,fmi}=cnn{i}.weights{plfmi,fmi}-(cnn{i}.deltaWeights{plfmi,fmi}+momentum.*preDeltaWeights{plfmi,fmi});% 调整权值
                        end
                    end
                end
            end
        end
    elseif strcmp(cnn{i}.layerType,'batchNormalizationLayer')% ------------BN层计算------------
        
        preLayerOutput=cnn{i-1}.output;% 获取上层输出
        beta=cnn{i}.beta;
        gamma=cnn{i}.gamma;
        epsilon=10^(-10);
        
        outputExpectation=cnn{i}.outputExpectation{batchNumI,1};% 取出上层输出均值(期望)
        outputVarience=cnn{i}.outputVarience{batchNumI,1};% 取出上层输出方差
        
        if (size(preLayerOutput{1,1},2)==1)&&(size(preLayerOutput{1,1},1)~=1)% 如果上一层是全连接层，则按全连接层的方式计算
            
            % ------计算上层误差------
            preLayerOutputMatrix=nan(batchSize,size(preLayerOutput{1},1));
            residual=nan(batchSize,size(preLayerOutput{1},1));
            for bs=1:batchSize% 依层计算批中的每个样本
                preLayerOutputMatrix(bs,:)=preLayerOutput{bs}';% 将上层输出转为矩阵
                residual(bs,:)=(cnn{i}.err{bs}').*actFun(cnn{i}.output{bs,1}','derivative');%计算本层残差：将网络输出端原始误差反向传播为输入端误差
            end
            
            % 此处参照《Batch Normalization：Accelerating Deep Network Training
            % by Reducing Internal Covariate Shift》中P4的链式法则公式
            temp1=residual.*repmat(gamma,batchSize,1);
            temp2=sum(temp1.*(preLayerOutputMatrix-repmat(outputExpectation,batchSize,1)),1).*((-1/2)*((outputVarience+epsilon).^(-3/2)));
            temp3=sum(temp1.*((-1)./sqrt(repmat(outputVarience+epsilon,size(preLayerOutputMatrix,1),1))),1)+temp2.*mean((-2)*(preLayerOutputMatrix-repmat(outputExpectation,batchSize,1)),1);
            preLayerErr=temp1.*(1./sqrt(repmat(outputVarience+epsilon,size(preLayerOutputMatrix,1),1)))+repmat(temp2,batchSize,1).*(2*(preLayerOutputMatrix-repmat(outputExpectation,batchSize,1))./batchSize)+repmat(temp3,batchSize,1)./batchSize;
            
            for bs=1:batchSize% 传播梯度给上一层
                cnn{i-1}.err{bs,1}=preLayerErr(bs,:)';
                %                 cnn{i-1}.err{bs,1}=zeros(size(preLayerErr(bs,:)'));
            end
            
            % ------调整本层权值与偏置------
            %             preLayerOutputMatrixNorm=(preLayerOutputMatrix-repmat(outputExpectation,batchSize,1))./sqrt(repmat(outputVarience,batchSize,1)+epsilon);
            
            preLayerOutputMatrixNorm=nan(batchSize,size(preLayerOutput{1},1));
            for bs=1:batchSize% 依层计算批中的每个样本
                preLayerOutputMatrixNorm(bs,:)=cnn{i}.preLayerOutputNorm{bs}';% 将上层标准化输出转为矩阵
            end
            
            deltaGamma=learningRate*sum(residual.*preLayerOutputMatrixNorm,1);
            deltaBeta=learningRate*sum(residual,1);
            cnn{i}.gamma=gamma-deltaGamma;
            cnn{i}.beta=beta-deltaBeta;
            
            %                         [mean(outputExpectation) mean(beta) mean(abs(outputExpectation-beta)) mean(deltaBeta)]
            %                         [mean(outputVarience) mean(gamma) mean(abs(outputVarience-gamma)) mean(deltaBeta)]
            %
            %                         [i (mean(gamma)./mean(sqrt(outputVarience+epsilon))) (-((mean(gamma).*mean(outputExpectation))./mean(sqrt(outputVarience+epsilon)))+mean(beta))]
            %
            %
            %                         tempOutput=zeros(size(cnn{i}.output,1),size(cnn{i}.output{1},1));
            %                         for tempOI=1:size(cnn{i}.output,1)
            %                             tempOutput(tempOI,:)=cnn{i}.output{tempOI,1}';
            %
            %                         end
            % %                         tempOutput
            %                         [mean(tempOutput,1);var(tempOutput)]
            %                         [mean(mean(tempOutput,1)) mean(var(tempOutput))]
            %                         pause
            
            
            
            %             pause
            %             deltaGamma
            %             beta
            %             outputExpectation
            %             sum(abs(((outputExpectation-beta)>0)-(deltaBeta>0)))
            
            
            %             gamma
            %             sqrt(outputVarience+epsilon)
            %             sum(abs(((sqrt(outputVarience+epsilon)-gamma)>0)-(deltaGamma>0)))
            
            %             pause
            
            
            
            
        else% 非全连接层的前一层为用cell包装的特征映射矩阵，为batchSize*featureMapNum
            % ------计算上层误差------
            err=cnn{i}.err;
            featureMapNum=size(preLayerOutput,2);
            featureMapSize=size(preLayerOutput{1,1},1);
            
            % 此处参照《Batch Normalization：Accelerating Deep Network Training
            % by Reducing Internal Covariate Shift》中P4的链式法则公式
            temp1=cell(batchSize,featureMapNum);
            residual=cell(batchSize,featureMapNum);
            for bs=1:batchSize
                for fmi=1:featureMapNum
                    residual(bs,fmi)={err{bs,fmi}.*actFun(cnn{i}.output{bs,fmi},'derivative')};
                    temp1{bs,fmi}=residual{bs,fmi}*gamma(1,fmi);
                end
            end
            
            
            temp2=zeros(1,featureMapNum);
            for fmi=1:featureMapNum
                for bs=1:batchSize
                    temp21=(temp1{bs,fmi}.*(preLayerOutput{bs,fmi}-outputExpectation(1,fmi)))*((-1/2)*((outputVarience(1,fmi)+epsilon).^(-3/2)));
                    temp2(1,fmi)=temp2(1,fmi)+sum(temp21(:));
                end
            end
            
            temp3=zeros(1,featureMapNum);
            temp31=zeros(1,featureMapNum);
            temp32=zeros(1,featureMapNum);
            for fmi=1:featureMapNum
                for bs=1:batchSize
                    temp31(1,fmi)=temp31(1,fmi)+(sum(temp1{bs,fmi}(:))*((-1)/sqrt(outputVarience(1,fmi)+epsilon)));
                    temp32(1,fmi)=temp32(1,fmi)+(-2)*sum(sum((preLayerOutput{bs,fmi}-outputExpectation(1,fmi))));
                    if(bs==batchSize)
                        temp3(1,fmi)=temp31(1,fmi)+temp2(1,fmi)*(temp32(1,fmi)/(batchSize*featureMapSize*featureMapSize));
                    end
                end
            end
            
            for fmi=1:featureMapNum
                for bs=1:batchSize
                    preLayerErrTemp1=(temp1{bs,fmi}*(1./sqrt(outputVarience(1,fmi)+epsilon)));
                    preLayerErrTemp2=(temp2(1,fmi)*(2*(preLayerOutput{bs,fmi}-outputExpectation(1,fmi))./(batchSize*featureMapSize*featureMapSize)));
                    preLayerErrTemp3=(temp3(1,fmi)./(batchSize*featureMapSize*featureMapSize));
                    %                     preLayerErrTemp3=(temp3(1,fmi)./batchSize);
                    
                    cnn{i-1}.err{bs,fmi}=(preLayerErrTemp1+preLayerErrTemp2+preLayerErrTemp3);% 传播梯度给上一层
                end
            end
            
            
            % ------调整本层权值与偏置------
            
            preLayerOutputNorm=cnn{i}.preLayerOutputNorm;
            
            
            
            deltaBeta=zeros(1,featureMapNum);
            deltaGamma=zeros(1,featureMapNum);
            for fmi=1:featureMapNum
                for bs=1:batchSize
                    %                     deltaBeta(1,fmi)=deltaBeta(1,fmi)+sum(sum(err{bs,fmi}.*preLayerOutputNorm{bs,fmi}))./(featureMapSize*featureMapSize);
                    %                     deltaGamma(1,fmi)=deltaGamma(1,fmi)+sum(sum(err{bs,fmi}))./(featureMapSize*featureMapSize);
                    deltaGamma(1,fmi)=deltaGamma(1,fmi)+sum(sum(residual{bs,fmi}.*preLayerOutputNorm{bs,fmi}));
                    deltaBeta(1,fmi)=deltaBeta(1,fmi)+sum(residual{bs,fmi}(:));
                end
            end
            deltaBeta=learningRate*deltaBeta;
            deltaGamma=learningRate*deltaGamma;
            %             deltaBeta
            %             deltaGamma
            %             pause
            
            cnn{i}.beta=beta-deltaBeta;
            cnn{i}.gamma=gamma-deltaGamma;
            
            
            
            %                         [mean(outputExpectation(:)) mean(beta(:)) mean(outputExpectation(:)-beta(:)) mean(deltaBeta(:))]
            %                         [mean(sqrt(outputVarience(1,fmi)+epsilon)) mean(gamma(:)) mean(sqrt(outputVarience(1,fmi)+epsilon)-gamma(:)) mean(deltaGamma(:))]
            %
            %                         [i (mean(gamma(:))./mean(sqrt(outputVarience(:)+epsilon))) (-((mean(gamma(:)).*mean(outputExpectation(:)))./mean(sqrt(outputVarience(:)+epsilon)))+mean(beta(:)))]
            %
            %
            %
            %                         tempInput=cell(1,size(cnn{i}.output,2));
            %                         meanInput=nan(1,size(cnn{i}.output,2));
            %                         meanVarInput=nan(1,size(cnn{i}.output,2));
            %                         for tempFMI=1:size(cnn{i}.output,2)
            % %                             tempFMO=zeros(size(cnn{i}.output{1,1}));
            %                             tempFMO=[];
            %                             for tempOI=1:size(cnn{i}.output,1)
            %                                 tempFMO=[tempFMO;cnn{i-1}.output{tempOI,tempFMI}];
            %                             end
            %                             tempInput{1,tempFMI}=tempFMO;
            %                             meanInput(1,tempFMI)=mean(tempInput{1,tempFMI}(:));
            %                             meanVarInput(tempFMI)=var(tempInput{1,tempFMI}(:));
            %                         end
            %
            %
            %                         [meanInput;meanVarInput]
            %                         tempOutput=cell(1,size(cnn{i}.output,2));
            %                         meanOutput=nan(1,size(cnn{i}.output,2));
            %                         meanVar=nan(1,size(cnn{i}.output,2));
            %                         for tempFMI=1:size(cnn{i}.output,2)
            % %                             tempFMO=zeros(size(cnn{i}.output{1,1}));
            %                             tempFMO=[];
            %                             for tempOI=1:size(cnn{i}.output,1)
            %                                 tempFMO=[tempFMO;cnn{i}.output{tempOI,tempFMI}];
            %                             end
            %                             tempOutput{1,tempFMI}=tempFMO;
            %                             meanOutput(1,tempFMI)=mean(tempOutput{1,tempFMI}(:));
            %                             meanVar(tempFMI)=var(tempOutput{1,tempFMI}(:));
            %                         end
            %
            %
            %                         [meanOutput;meanVar]
            
            %                         [mean(tempOutput,1);var(tempOutput)]
            %                         [mean(mean(tempOutput,1)) mean(var(tempOutput))]
            
            
            %
            
            
            
        end
    elseif strcmp(cnn{i}.layerType,'dropoutLayer')% ------------dropout层计算------------
        
        if isfield(cnn{i-1},'err')==1% 只有非输入层才需要计算上层误差
            if(cnn{i}.dropoutRate>0)%如果dropoutRate==0则不进行dropout
                for bs=1:batchSize% 对本batch中的每一个样本
                    if (size(cnn{i-1}.output{1,1},2)==1)&&(size(cnn{i-1}.output{1,1},1)~=1)% 如果上一层是全连接层输出格式，则按全连接层的方式计算
                        
                        cnn{i-1}.err{bs,1}=cnn{i}.err{bs,1}.*cnn{i}.dropoutMask{bs,1};% 对当前样本输出做dropout
                        
                    else% 非全连接层的前一层为用cell包装的特征映射矩阵，为batchSize*featureMapNum
                        featureMapNum=size(cnn{i-1}.output,2);
                        
                        for fmi=1:featureMapNum
                            cnn{i-1}.err{bs,fmi}=cnn{i}.err{bs,fmi}.*cnn{i}.dropoutMask{bs,fmi};% 对当前样本输出做dropout
                            
                        end
                    end
                end
            else
                cnn{i-1}.err=cnn{i}.err;
            end
        end
    end
    
    
    if isfield(cnn{i},'weights')
        if strcmp(regularizationType,'nan')
            if (size(cnn{i}.output{1,1},2)==1)&&(size(cnn{i}.output{1,1},1)~=1)% 如果本层是全连接层，则按全连接层的方式计算
                cnn{i}.weights{1}=min(cnn{i}.weights{1},cnn{i}.maxWeightRange);
            else
                for fmi=1:size(cnn{i}.weights,2)% 调整卷积核的权值和偏置值
                    for plfmi=1:size(cnn{i}.weights,1)% 将每个上层特征映射矩阵对应的所有本层误差矩阵过一次
                        cnn{i}.weights{plfmi,fmi}=min(cnn{i}.weights{plfmi,fmi},cnn{i}.maxWeightRange);
                    end
                end
            end
        elseif strcmp(regularizationType,'L2')
            if (size(cnn{i}.output{1,1},2)==1)&&(size(cnn{i}.output{1,1},1)~=1)% 如果本层是全连接层，则按全连接层的方式计算
                cnn{i}.weights{1}=cnn{i}.weights{1}-(regularizationLamda/trainSetNum)*learningRate*regularWeights{1};
                cnn{i}.weights{1}=min(cnn{i}.weights{1},cnn{i}.maxWeightRange);
            else
                for fmi=1:size(cnn{i}.weights,2)% 调整卷积核的权值和偏置值
                    for plfmi=1:size(cnn{i}.weights,1)% 将每个上层特征映射矩阵对应的所有本层误差矩阵过一次
                        cnn{i}.weights{plfmi,fmi}=cnn{i}.weights{plfmi,fmi}-(regularizationLamda/trainSetNum)*learningRate*regularWeights{plfmi,fmi};
                        cnn{i}.weights{plfmi,fmi}=min(cnn{i}.weights{plfmi,fmi},cnn{i}.maxWeightRange);
                    end
                end
            end
        elseif strcmp(regularizationType,'L1')
            if (size(cnn{i}.output{1,1},2)==1)&&(size(cnn{i}.output{1,1},1)~=1)% 如果本层是全连接层，则按全连接层的方式计算
                sign=((regularWeights{1}>0)+(-1)*(regularWeights{1}<0));
                cnn{i}.weights{1}=cnn{i}.weights{1}-(regularizationLamda/trainSetNum)*learningRate*sign;
                cnn{i}.weights{1}=min(cnn{i}.weights{1},cnn{i}.maxWeightRange);
            else
                for fmi=1:size(cnn{i}.weights,2)% 调整卷积核的权值和偏置值
                    for plfmi=1:size(cnn{i}.weights,1)% 将每个上层特征映射矩阵对应的所有本层误差矩阵过一次
                        sign=((cnn{i}.weights{plfmi,fmi}>0)+(-1)*(cnn{i}.weights{plfmi,fmi}<0));
                        cnn{i}.weights{plfmi,fmi}=cnn{i}.weights{plfmi,fmi}-(regularizationLamda/trainSetNum)*learningRate*sign;
                        cnn{i}.weights{plfmi,fmi}=min(cnn{i}.weights{plfmi,fmi},cnn{i}.maxWeightRange);
                    end
                end
            end
        end
    end
    
    
    
    
    
    
    %     cnn{i}.err{1,1}
    %     if i==2 || i==4 || i==7
    %         cnn{i}.weights{1,1}
    %         cnn{i}.deltaWeights{1,1}
    %     end
    %
    %     [batchNumI,i]
    %
    % %     pause(0.5)
    %     pause
    
end








end