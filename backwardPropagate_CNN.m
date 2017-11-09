function cnn=backwardPropagate_CNN(cnn,err_Output,batchNumI,epoch,trainSetNum,regularizationLamda,regularizationType)
batchSize=size(err_Output,1);% ������������
cnn{end}.err=err_Output;% ���������ԭʼ����ʼ��

for i=length(cnn):-1:2% ���ݽṹ��������Ӧ�������ÿһ������
    
    if isfield(cnn{i},'actFun')
        if strcmp(cnn{i}.actFun,'emptyFun')% ��ȡ��ǰ���Ӧ�������Ӧ�ķ�����
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
        momentum=cnn{i}.momentum;% ��ȡ�������
    end
    
    if isfield(cnn{i},'learningRate')
        learningRate=cnn{i}.learningRate;% ��ȡ����ѧϰ��
    end
    
    decayEpoch=30;
    %     decayRate=0.5;
    decayRate=1;
    if(epoch<decayEpoch&&(~strcmp(cnn{i}.layerType,'batchNormalizationLayer')))% ��BN���ѧϰ����epoch��ν���
        learningRate=learningRate*(decayRate^epoch);
    else
        learningRate=learningRate*(decayRate^decayEpoch);
    end
    
    regularWeights=nan;
    
    
    if strcmp(cnn{i}.layerType,'fullConnLayer')% ------------ȫ���Ӳ����------------
        residual=cell(batchSize,1);
        for bs=1:batchSize% ����������е�ÿ������
            % ------�����ϲ����------
            weights=cnn{i}.weights{1};% ��ȡȨֵ��Ϊһ����
            err=cnn{i}.err{bs,1};% ��ȡ��ǰ��ԭʼ��Ϊһ����
            residual{bs,1}=err.*actFun(cnn{i}.output{bs,1},'derivative');%���㱾��в�����������ԭʼ���򴫲�Ϊ��������
            
            if isfield(cnn{i-1},'err')==1% ֻ�з���������Ҫ�����ϲ����
                perLayerErr=residual{bs}'*weights';% ����ǰһ�������ԭʼ��Ϊһ����
                perLayerErr=perLayerErr';% ��������תΪ������
                size1=length(perLayerErr);% ǰһ�������������
                size2=size(cnn{i-1}.err,2);% ǰһ��������ĸ���
                size3=size1/size2;% ����ÿ��������Ҫ��ȡ�������������Ϊ����
                
                if (size(cnn{i-1}.output{1,1},2)==1)&&(size(cnn{i-1}.output{1,1},1)~=1)% �����һ����ȫ���Ӳ������ʽ
                    cnn{i-1}.err{bs,1}=perLayerErr;
                else
                    for ei=1:size2
                        tempErrMap=perLayerErr((ei-1)*size3+1:ei*size3);% ��ȡ��ei���������Ӧ���������
                        tempErrMap=reshape(tempErrMap,sqrt(size3),sqrt(size3));% ����ת��Ϊ���󣬰��е�˳������
                        cnn{i-1}.err{bs,ei}=tempErrMap';% ת��Ϊ�������в��洢���ϲ������
                    end
                end
            end
            % ------��������Ȩֵ��ƫ��------
            regularWeights=cnn{i}.weights;
            % ��ɺ����һ�����
            preLayerOutput=cnn{i-1}.output(bs,:);% ��һ�������һά��������󼯣�
            preLayerOutput1{bs}=[];% ��ɺ������
            preOutputSize=length(preLayerOutput{1,1});% �ϲ�����ӳ��ĳߴ�
            for i1=1:length(preLayerOutput)% ��ɺ��������һ����������ӳ���������һ�в���������
                for i2=1:preOutputSize
                    preLayerOutput1{bs}=[preLayerOutput1{bs},preLayerOutput{i1}(i2,:)];
                end
            end
            
            if bs==batchSize% �ڼ�����ÿ�����������ͨ������ƽ��ֵ�������������
                
                
                
                renewBias=0;
                if i<length(cnn)% �ж���Ҫ����Ҫ����bias�������һ�㲻��BN������£���renewBias=1;����Ϊ0
                    if strcmp(cnn{i+1}.layerType,'batchNormalizationLayer')==0
                        renewBias=1;
                    end
                elseif i==length(cnn)
                    renewBias=1;
                end
                
                
                
                preDeltaWeights=cnn{i}.deltaWeights{1};% ��ȡǰһ��Ȩֵ�ı���
                
                deltaWeightsAvg=zeros(size(residual{1},1),size(preLayerOutput1{1},2));
                residualAvg=zeros(size(residual{1}));
                
                for bs1=1:batchSize% �������в�����Ȩֵ�ı�����ƽ��ֵ
                    deltaWeightsAvg=deltaWeightsAvg+residual{bs1}*preLayerOutput1{bs1};% ����Ȩֵ�ı���ƽ��ֵ
                    residualAvg=residualAvg+residual{bs1};% ����в��ƽ��ֵ
                end
                deltaWeightsAvg=deltaWeightsAvg./batchSize;
                residualAvg=residualAvg./batchSize;
                
                cnn{i}.deltaWeights{1}=learningRate.*(deltaWeightsAvg)';% ����Ȩֵ�ı���
                cnn{i}.deltaBias{1}=learningRate.*residualAvg;% ������ֵ�ı���
                cnn{i}.weights{1}=cnn{i}.weights{1}-(cnn{i}.deltaWeights{1}+momentum.*preDeltaWeights);% ����Ȩֵ
                
                if renewBias==1
                    preDeltaBias=cnn{i}.deltaBias{1};% ��ȡǰһ����ֵ�ı���
                    cnn{i}.bias{1}=cnn{i}.bias{1}-(cnn{i}.deltaBias{1}+momentum.*preDeltaBias);% ����ƫ��
                end
            end
        end
    elseif strcmp(cnn{i}.layerType,'subSampLayer')% ------------�����������------------
        
        for bs=1:batchSize% ����������е�ÿ������
            
            if isfield(cnn{i-1},'err')==1% ֻ�з���������Ҫ�����ϲ����
                % ------�����ϲ����------
                
                preLayerErrSize=length(cnn{i-1}.err{bs,1});% �ϲ�������ߴ�
                featuresMapNum=size(cnn{i}.output,2);% ����ӳ�������������
                
                for fmi=1:featuresMapNum
                    expandedErr{fmi}=kron(cnn{i}.err{bs,fmi},ones(cnn{i}.size));% ��չ��ı��������ԭʼ�����󣬳ߴ���ڻ����ǰ��������
                end
                
                if cnn{i}.isDirConn==1% ���Ϊֱ������в���������ԭʼ������Ҫͨ��������ķ��������������м���
                    for fmi=1:featuresMapNum% �����ϲ�ÿ��������
                        cnn{i-1}.err{bs,fmi}=expandedErr{fmi}(1:preLayerErrSize,1:preLayerErrSize);
                    end
                else% ���Ϊ��ֱ��������Ҫ���ϲ�������з��򴫲�����������
                    
                end
                
            end
            % ------��������Ȩֵ��ƫ��------
            
            
            % ��ֱ������Ҫ�������˴���
        end
    elseif strcmp(cnn{i}.layerType,'convLayer')% ------------��������------------
        featuresMapNum=size(cnn{i}.output,2);% ����ӳ�������������
        residual=cell(batchSize,featuresMapNum);
        for bs=1:batchSize% ����������е�ÿ������
            % ------�����ϲ����------
            if strcmp(cnn{i}.connType,'local')
                connIndexMatrix=cnn{i}.connIndexMatrix;
            end
            featuresMapNum=size(cnn{i}.err,2);% ��������������
            preLayerFeaturesMapNum=size(cnn{i-1}.output,2);% �ϲ�����ӳ�������������
            preLayerErrSize=size(cnn{i-1}.output{bs,1},1);% �ϲ�����ӳ���������ߴ�
            preLayerOutput{bs}=cnn{i-1}.output(bs,:);% ��һ������������󼯣�
            for fmi=1:featuresMapNum% ���㱾������˲в�
                residual(bs,fmi)={cnn{i}.err{bs,fmi}.*actFun(cnn{i}.output{bs,fmi},'derivative')};
            end
            if isfield(cnn{i-1},'err')==1% ֻ�з���������Ҫ�����ϲ����
                for plfmi=1:preLayerFeaturesMapNum% ��ÿ���ϲ�����ӳ������Ӧ�����б����������һ��
                    sumErr=zeros(preLayerErrSize);
                    for fmi=1:featuresMapNum% �ۼ�ÿ���ϲ���������ÿ���������
                        if strcmp(cnn{i}.connType,'local')
                            if connIndexMatrix(fmi,plfmi)~=0
                                sumErr=sumErr+conv2(residual{bs,fmi},cnn{i}.weights{plfmi,fmi},'full');
                            end
                        elseif strcmp(cnn{i}.connType,'full')
                            sumErr=sumErr+conv2(residual{bs,fmi},cnn{i}.weights{plfmi,fmi},'full');
                        end
                    end
                    cnn{i-1}.err{bs,plfmi}=sumErr;% �����ϲ�ÿ�������ԭʼ�������ֵ
                end
            end
            
            % ------��������Ȩֵ��ƫ��------
            regularWeights=cnn{i}.weights;
            if bs==batchSize% �ڼ�����ÿ�����������ͨ������ƽ��ֵ�������������
                
                
                
                renewBias=0;
                if i<length(cnn)% �ж���Ҫ����Ҫ����bias�������һ�㲻��BN������£���renewBias=1;����Ϊ0
                    if strcmp(cnn{i+1}.layerType,'batchNormalizationLayer')==0
                        renewBias=1;
                    end
                elseif i==length(cnn)
                    renewBias=1;
                end
                
                
                
                if renewBias==1
                    deltaBiasAvg=cell(1,featuresMapNum);% �в����ƽ��ֵ
                    for fmi=1:featuresMapNum% �������ӳ��������
                        deltaBiasAvg{fmi}=zeros(size(residual{1,1}));
                        for bs1=1:batchSize% �ۼӱ����и�����в�����ֵ������ƽ��ֵ
                            deltaBiasAvg{fmi}=deltaBiasAvg{fmi}+(learningRate).*residual{bs1,fmi};% �ۼ�������������ֵ�ı���
                            if bs1==batchSize
                                deltaBiasAvg{fmi}=deltaBiasAvg{fmi}./batchSize;% ��ÿ��ƫ�þ���ĸı���ƽ��ֵ
                            end
                        end
                    end
                    preDeltaBias=cnn{i}.deltaBias;% ��ȡǰһ����ֵ�ı���
                    cnn{i}.deltaBias=deltaBiasAvg;% �ñ���ƽ���в���µ�ǰ�Ĳв������
                end
                
                
                deltaWeightsAvg=cell(preLayerFeaturesMapNum,featuresMapNum);% Ȩֵ����ı���ƽ��ֵ
                
                for plfmi=1:preLayerFeaturesMapNum% ��ÿ���ϲ�����ӳ������Ӧ�����б����������һ��
                    for fmi=1:featuresMapNum% �������ӳ��������
                        deltaWeightsAvg{plfmi,fmi}=zeros(size(cnn{i}.weights{1,1}));
                        for bs1=1:batchSize% �ۼӱ����и�������Ӧ��Ȩֵ�ı����ĸı���������ƽ��ֵ
                            if strcmp(cnn{i}.connType,'local')
                                if connIndexMatrix(fmi,plfmi)~=0
                                    deltaWeightsAvg{plfmi,fmi}=deltaWeightsAvg{plfmi,fmi}+(learningRate).*conv2(preLayerOutput{bs1}{plfmi},residual{bs1,fmi}(end:-1:1,end:-1:1),'valid');% �ۼ�����������Ȩֵ�ı���
                                end
                            elseif strcmp(cnn{i}.connType,'full')
                                deltaWeightsAvg{plfmi,fmi}=deltaWeightsAvg{plfmi,fmi}+(learningRate).*conv2(preLayerOutput{bs1}{plfmi},residual{bs1,fmi}(end:-1:1,end:-1:1),'valid');% �ۼ�����������Ȩֵ�ı���
                            end
                            if bs1==batchSize
                                deltaWeightsAvg{plfmi,fmi}=deltaWeightsAvg{plfmi,fmi}./batchSize;% ��ÿ��Ȩֵ����ĸı���ƽ��ֵ
                            end
                        end
                    end
                end
                
                preDeltaWeights=cnn{i}.deltaWeights;
                cnn{i}.deltaWeights=deltaWeightsAvg;
                
                for fmi=1:featuresMapNum% ��������˵�Ȩֵ��ƫ��ֵ
                    
                    if renewBias==1
                        cnn{i}.bias{fmi}=cnn{i}.bias{fmi}+cnn{i}.deltaBias{fmi}+momentum.*preDeltaBias{fmi};% �����������ӳ���Ӧ��ƫ��
                    end
                    for plfmi=1:preLayerFeaturesMapNum% ��ÿ���ϲ�����ӳ������Ӧ�����б����������һ��
                        if strcmp(cnn{i}.connType,'local')
                            if connIndexMatrix(fmi,plfmi)~=0
                                cnn{i}.weights{plfmi,fmi}=cnn{i}.weights{plfmi,fmi}-(cnn{i}.deltaWeights{plfmi,fmi}+momentum.*preDeltaWeights{plfmi,fmi});% ����Ȩֵ
                            end
                        elseif strcmp(cnn{i}.connType,'full')
                            cnn{i}.weights{plfmi,fmi}=cnn{i}.weights{plfmi,fmi}-(cnn{i}.deltaWeights{plfmi,fmi}+momentum.*preDeltaWeights{plfmi,fmi});% ����Ȩֵ
                        end
                    end
                end
            end
        end
    elseif strcmp(cnn{i}.layerType,'batchNormalizationLayer')% ------------BN�����------------
        
        preLayerOutput=cnn{i-1}.output;% ��ȡ�ϲ����
        beta=cnn{i}.beta;
        gamma=cnn{i}.gamma;
        epsilon=10^(-10);
        
        outputExpectation=cnn{i}.outputExpectation{batchNumI,1};% ȡ���ϲ������ֵ(����)
        outputVarience=cnn{i}.outputVarience{batchNumI,1};% ȡ���ϲ��������
        
        if (size(preLayerOutput{1,1},2)==1)&&(size(preLayerOutput{1,1},1)~=1)% �����һ����ȫ���Ӳ㣬��ȫ���Ӳ�ķ�ʽ����
            
            % ------�����ϲ����------
            preLayerOutputMatrix=nan(batchSize,size(preLayerOutput{1},1));
            residual=nan(batchSize,size(preLayerOutput{1},1));
            for bs=1:batchSize% ����������е�ÿ������
                preLayerOutputMatrix(bs,:)=preLayerOutput{bs}';% ���ϲ����תΪ����
                residual(bs,:)=(cnn{i}.err{bs}').*actFun(cnn{i}.output{bs,1}','derivative');%���㱾��в�����������ԭʼ���򴫲�Ϊ��������
            end
            
            % �˴����ա�Batch Normalization��Accelerating Deep Network Training
            % by Reducing Internal Covariate Shift����P4����ʽ����ʽ
            temp1=residual.*repmat(gamma,batchSize,1);
            temp2=sum(temp1.*(preLayerOutputMatrix-repmat(outputExpectation,batchSize,1)),1).*((-1/2)*((outputVarience+epsilon).^(-3/2)));
            temp3=sum(temp1.*((-1)./sqrt(repmat(outputVarience+epsilon,size(preLayerOutputMatrix,1),1))),1)+temp2.*mean((-2)*(preLayerOutputMatrix-repmat(outputExpectation,batchSize,1)),1);
            preLayerErr=temp1.*(1./sqrt(repmat(outputVarience+epsilon,size(preLayerOutputMatrix,1),1)))+repmat(temp2,batchSize,1).*(2*(preLayerOutputMatrix-repmat(outputExpectation,batchSize,1))./batchSize)+repmat(temp3,batchSize,1)./batchSize;
            
            for bs=1:batchSize% �����ݶȸ���һ��
                cnn{i-1}.err{bs,1}=preLayerErr(bs,:)';
                %                 cnn{i-1}.err{bs,1}=zeros(size(preLayerErr(bs,:)'));
            end
            
            % ------��������Ȩֵ��ƫ��------
            %             preLayerOutputMatrixNorm=(preLayerOutputMatrix-repmat(outputExpectation,batchSize,1))./sqrt(repmat(outputVarience,batchSize,1)+epsilon);
            
            preLayerOutputMatrixNorm=nan(batchSize,size(preLayerOutput{1},1));
            for bs=1:batchSize% ����������е�ÿ������
                preLayerOutputMatrixNorm(bs,:)=cnn{i}.preLayerOutputNorm{bs}';% ���ϲ��׼�����תΪ����
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
            
            
            
            
        else% ��ȫ���Ӳ��ǰһ��Ϊ��cell��װ������ӳ�����ΪbatchSize*featureMapNum
            % ------�����ϲ����------
            err=cnn{i}.err;
            featureMapNum=size(preLayerOutput,2);
            featureMapSize=size(preLayerOutput{1,1},1);
            
            % �˴����ա�Batch Normalization��Accelerating Deep Network Training
            % by Reducing Internal Covariate Shift����P4����ʽ����ʽ
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
                    
                    cnn{i-1}.err{bs,fmi}=(preLayerErrTemp1+preLayerErrTemp2+preLayerErrTemp3);% �����ݶȸ���һ��
                end
            end
            
            
            % ------��������Ȩֵ��ƫ��------
            
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
    elseif strcmp(cnn{i}.layerType,'dropoutLayer')% ------------dropout�����------------
        
        if isfield(cnn{i-1},'err')==1% ֻ�з���������Ҫ�����ϲ����
            if(cnn{i}.dropoutRate>0)%���dropoutRate==0�򲻽���dropout
                for bs=1:batchSize% �Ա�batch�е�ÿһ������
                    if (size(cnn{i-1}.output{1,1},2)==1)&&(size(cnn{i-1}.output{1,1},1)~=1)% �����һ����ȫ���Ӳ������ʽ����ȫ���Ӳ�ķ�ʽ����
                        
                        cnn{i-1}.err{bs,1}=cnn{i}.err{bs,1}.*cnn{i}.dropoutMask{bs,1};% �Ե�ǰ���������dropout
                        
                    else% ��ȫ���Ӳ��ǰһ��Ϊ��cell��װ������ӳ�����ΪbatchSize*featureMapNum
                        featureMapNum=size(cnn{i-1}.output,2);
                        
                        for fmi=1:featureMapNum
                            cnn{i-1}.err{bs,fmi}=cnn{i}.err{bs,fmi}.*cnn{i}.dropoutMask{bs,fmi};% �Ե�ǰ���������dropout
                            
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
            if (size(cnn{i}.output{1,1},2)==1)&&(size(cnn{i}.output{1,1},1)~=1)% ���������ȫ���Ӳ㣬��ȫ���Ӳ�ķ�ʽ����
                cnn{i}.weights{1}=min(cnn{i}.weights{1},cnn{i}.maxWeightRange);
            else
                for fmi=1:size(cnn{i}.weights,2)% ��������˵�Ȩֵ��ƫ��ֵ
                    for plfmi=1:size(cnn{i}.weights,1)% ��ÿ���ϲ�����ӳ������Ӧ�����б����������һ��
                        cnn{i}.weights{plfmi,fmi}=min(cnn{i}.weights{plfmi,fmi},cnn{i}.maxWeightRange);
                    end
                end
            end
        elseif strcmp(regularizationType,'L2')
            if (size(cnn{i}.output{1,1},2)==1)&&(size(cnn{i}.output{1,1},1)~=1)% ���������ȫ���Ӳ㣬��ȫ���Ӳ�ķ�ʽ����
                cnn{i}.weights{1}=cnn{i}.weights{1}-(regularizationLamda/trainSetNum)*learningRate*regularWeights{1};
                cnn{i}.weights{1}=min(cnn{i}.weights{1},cnn{i}.maxWeightRange);
            else
                for fmi=1:size(cnn{i}.weights,2)% ��������˵�Ȩֵ��ƫ��ֵ
                    for plfmi=1:size(cnn{i}.weights,1)% ��ÿ���ϲ�����ӳ������Ӧ�����б����������һ��
                        cnn{i}.weights{plfmi,fmi}=cnn{i}.weights{plfmi,fmi}-(regularizationLamda/trainSetNum)*learningRate*regularWeights{plfmi,fmi};
                        cnn{i}.weights{plfmi,fmi}=min(cnn{i}.weights{plfmi,fmi},cnn{i}.maxWeightRange);
                    end
                end
            end
        elseif strcmp(regularizationType,'L1')
            if (size(cnn{i}.output{1,1},2)==1)&&(size(cnn{i}.output{1,1},1)~=1)% ���������ȫ���Ӳ㣬��ȫ���Ӳ�ķ�ʽ����
                sign=((regularWeights{1}>0)+(-1)*(regularWeights{1}<0));
                cnn{i}.weights{1}=cnn{i}.weights{1}-(regularizationLamda/trainSetNum)*learningRate*sign;
                cnn{i}.weights{1}=min(cnn{i}.weights{1},cnn{i}.maxWeightRange);
            else
                for fmi=1:size(cnn{i}.weights,2)% ��������˵�Ȩֵ��ƫ��ֵ
                    for plfmi=1:size(cnn{i}.weights,1)% ��ÿ���ϲ�����ӳ������Ӧ�����б����������һ��
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