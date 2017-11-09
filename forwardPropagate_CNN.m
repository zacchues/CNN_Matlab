function [res,cnn]=forwardPropagate_CNN(cnn,input,batchNumI,batchNum,singleStepIndex)
batchSize=size(input,1);% ������������

if nargin==4
    beginI=1;
    endI=length(cnn);
elseif nargin==5
    beginI=singleStepIndex;
    endI=singleStepIndex;
end
for i=beginI:endI% ���ݽṹ��������Ӧ�������ÿһ������
    if isfield(cnn{i},'actFun')
        if strcmp(cnn{i}.actFun,'emptyFun')% ��ȡ��ǰ���Ӧ�����
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
    
    if strcmp(cnn{i}.layerType,'inputLayer')% ------------�����������------------
        cnn{i}.output(1:batchSize,:)=input;% ���������ֱ�ӵ���������ܵ���������
        
    elseif strcmp(cnn{i}.layerType,'convLayer')% ------------�����������------------
        for bs=1:batchSize% ����������е�ÿ������
            if strcmp(cnn{i}.connType,'full')% ���Ϊȫ����ʽ����ǰ�����е�����ǰ����������ӳ������
                
                convKernel=cnn{i}.weights;% ��ȡ��������
                for li=1:cnn{i}.featureMapNum% ��ÿ�����������ӳ��
                    tempOutput=zeros(cnn{i}.outputFMSize);
                    for pli=1:size(cnn{i-1}.output,2)% ��ÿ���ϲ������ӳ��
                        % ʹ��matlab�Դ��ľ�����������Զ�������˽���180����ת����������ʱ��Ҫ�˹���תһ����֮�����
                        tempOutput=tempOutput+convn(cnn{i-1}.output{bs,pli},convKernel{pli,li}(end:-1:1,end:-1:1),'valid');
                    end
                    cnn{i}.output{bs,li}=actFun(tempOutput+cnn{i}.bias{li});
                end
            elseif strcmp(cnn{i}.connType,'local')% ���Ϊ�ֲ�����ʽ����ǰ�����е�����ǰ����������ӳ������
                
                connIndexMatrix=cnn{i}.connIndexMatrix;
                
                convKernel=cnn{i}.weights;% ��ȡ��������
                
                for li=1:size(cnn{i}.output,2)% ��ÿ�����������ӳ��
                    tempOutput=zeros(cnn{i}.outputFMSize);
                    for pli=1:size(cnn{i-1}.output,2)% ��ÿ���ϲ������ӳ��
                        if connIndexMatrix(li,pli)~=0
                            % ʹ��matlab�Դ��ľ�����������Զ�������˽���180����ת����������ʱ��Ҫ�˹���תһ����֮�����
                            tempOutput=tempOutput+convn(cnn{i-1}.output{bs,pli},convKernel{pli,li}(end:-1:1,end:-1:1),'valid');
                        end
                    end
                    cnn{i}.output{bs,li}=actFun(tempOutput+cnn{i}.bias{li});
                end
            end
        end
        
    elseif strcmp(cnn{i}.layerType,'subSampLayer')% ------------��������������------------
        for bs=1:batchSize% ����������е�ÿ������
            featuresMapNum=size(cnn{i}.output,2);% ����ӳ������
            
            [l1,l2]=size(cnn{i}.output{1,1});% ��������ӳ��ߴ�
            sampSize=cnn{i}.size;% ��ȡ�����ֱ���
            perLayerOutputSize=size(cnn{i-1}.output{1,1},1);% ��ȡ�ϲ�����ӳ��ߴ�
            
            for fm=1:featuresMapNum% �Ա������ϲ��ÿ����Ӧ����ӳ��
                tempFM=cnn{i-1}.output{bs,fm};
                
                if strcmp(cnn{i}.sampType,'max')% ���ݲ�����ʽ���ɵ�ǰ���Ԫ�ص�ֵ
                    for i1=1:l1
                        for i2=1:l2
                            % ������ֱ������ϲ�����ӳ��ߴ粻ƥ�䣬��ʣ�µĲ�����Ϊ����ȫ���������������ڷ��򴫲�ʱ��Ҫ���ж��Ƿ���������Ѱ�Ҷ�Ӧ�ı�
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
                            sampRegion=tempFM(((i1-1)*sampSize+1):upIndex1,((i2-1)*sampSize+1):upIndex2);% ���ϲ�ȡ���������򱣴���sampRegion��
                            cnn{i}.output{bs,fm}(i1,i2)=max(sampRegion(:));
                        end
                    end
                elseif strcmp(cnn{i}.sampType,'mean')
                    for i1=1:l1
                        for i2=1:l2
                            % ������ֱ������ϲ�����ӳ��ߴ粻ƥ�䣬��ʣ�µĲ�����Ϊ����ȫ���������������ڷ��򴫲�ʱ��Ҫ���ж��Ƿ���������Ѱ�Ҷ�Ӧ�ı�
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
                            sampRegion=tempFM(((i1-1)*sampSize+1):upIndex1,((i2-1)*sampSize+1):upIndex2);% ���ϲ�ȡ���������򱣴���sampRegion��
                            cnn{i}.output{bs,fm}(i1,i2)=mean(sampRegion(:));
                        end
                    end
                end
                
                if cnn{i}.isDirConn==0% �����Ϊֱ������ʹ��ƫ�á�Ȩֵ����������������������
                end
            end
        end
    elseif strcmp(cnn{i}.layerType,'fullConnLayer')% ------------ȫ���Ӳ�������------------
        for bs=1:batchSize% ����������е�ÿ������
            % ��ɺ����һ�����
            preLayerOutput=cnn{i-1}.output(bs,:);% ��һ�������һά��������󼯣�
            
            preLayerOutput1=[];% ��ɺ������
            preOutputSize=size(preLayerOutput{1},1);% �ϲ�����ӳ��ĳߴ�
            
            for i1=1:length(preLayerOutput)% ��ɺ��������һ����������ӳ���������һ�в���������
                for i2=1:preOutputSize
                    preLayerOutput1=[preLayerOutput1,preLayerOutput{i1}(i2,:)];
                end
            end
            preLayerOutput1=preLayerOutput1';% ���תΪ������
            
            cnn{i}.output{bs,1}=actFun([preLayerOutput1'*cnn{i}.weights{1}]'+cnn{i}.bias{1});%�ü������Ȩֵ��ƫ�ô����ɺ���������
        end
    elseif strcmp(cnn{i}.layerType,'batchNormalizationLayer')% ------------ȫ���Ӳ�������------------
        preLayerOutput=cnn{i-1}.output;%��ȡ��һ���������
        beta=cnn{i}.beta;
        gamma=cnn{i}.gamma;
        epsilon=10^(-10);
        if (size(preLayerOutput{1,1},2)==1)&&(size(preLayerOutput{1,1},1)~=1)% �����һ����ȫ���Ӳ������ʽ����ȫ���Ӳ�ķ�ʽ����
            
            % ȫ���Ӳ����preLayerOutputΪsize*1��������������sizeΪȫ���Ӳ���Ԫ��
            
            output1=nan(batchSize,size(preLayerOutput{1},1));
            
            for bs=1:batchSize% ����������е�ÿ������
                output1(bs,:)=preLayerOutput{bs}';% ���ϲ����תΪ����
            end
            if (batchNum~=0)% batchNum~=0˵����ʱ��ѵ�����̣���Ҫ���㵱ǰ���ε������뷽��
                outputExpectation=mean(output1,1);% �ϲ������ֵ(����)
                if (size(output1,1)==1)% �ϲ��������
                    outputVarience=ones(1,size(output1,2));% �������ֻ��һ����������Ĭ�Ϸ���Ϊ1
                else
                    outputVarience=var(output1);% �����м��㷽��,���س���Ϊ1*size��������
                end
            else% batchNum==0˵����ʱ��Ԥ����̣�ֱ�ӵ�����ǰ������������ε������뷽������ƽ��ֵ
                outputExpectation=cnn{i}.outputExpectationAvg{1};
                outputVarience=cnn{i}.outputVarianceAvg{1};
            end
            preLayerOutputMatrixNorm=(output1-repmat(outputExpectation,batchSize,1))./repmat((sqrt(outputVarience)+epsilon),batchSize,1);% ��׼��ǰһ�����
            act=preLayerOutputMatrixNorm.*repmat(gamma,batchSize,1)+repmat(beta,batchSize,1);
            
            
            %             if batchNum~=0
            %             mean(act)
            %             var(act)
            %             pause
            %             end
            
            
            
            output1=actFun(act);% ��ǰһ������������λ�Ʋ��ü��������
            %             output1=actFun(preLayerOutputMatrixNorm);% ��ǰһ������������λ�Ʋ��ü��������
            
            
            for bs=1:batchSize% ��ԭΪԭʼ�����ʽ
                cnn{i}.preLayerOutputNorm{bs}=preLayerOutputMatrixNorm(bs,:)';
                cnn{i}.output{bs}=output1(bs,:)';
            end
            
        else% ��ȫ���Ӳ��ǰһ�����Ϊ��cell��װ������ӳ�����ΪbatchSize*featureMapNum
            
            % ��ȫ���Ӳ����preLayerOutputΪbatchSize*featureMapNum��cell������ÿ��cell�е�Ԫ��Ϊһ������ӳ�����
            
            featureMapNum=size(preLayerOutput,2);
            
            if (batchNum~=0)% batchNum~=0˵����ʱ��ѵ�����̣���Ҫ���㵱ǰ���εĸ�������ӳ�����������뷽��
                % ��ȫ���Ӳ���ϲ���������뷽�����batchSize*featureMapNum�ľ������洢
                outputExpectation=zeros(1,featureMapNum);
                outputVarience=zeros(1,featureMapNum);
                for bs=1:batchSize
                    for fmi=1:featureMapNum
                        outputExpectation(1,fmi)=outputExpectation(1,fmi)+mean(preLayerOutput{bs,fmi}(:));% �ϲ����������ӳ�����������Ԫ�صľ�ֵ(����)
                        % ע�⣺�˴�ʹ��var������ķ���Ϊ��������ֵ����ƫ����ֵ���˴�Ϊͼʡ�¾�ֱ������ƫ��������ʵֵ�˰ɣ�����ûʲô̫��Ӱ��
                        outputVarience(1,fmi)=outputVarience(1,fmi)+var(preLayerOutput{bs,fmi}(:));% �ϲ����������ӳ�����������Ԫ�صķ���(��ƫ)
                    end
                    if bs==batchSize
                        outputExpectation=outputExpectation./(batchSize);
                        outputVarience=outputVarience./(batchSize);
                    end
                end
                if(sum(size(preLayerOutput{1,1}))==2)% ����ϲ����Ϊֻ��һ��Ԫ�ص�����ӳ��ͼ����Ĭ�Ϸ���Ϊ1
                    outputVarience=ones(1,featureMapNum);
                end
            else% batchNum==0˵����ʱ��Ԥ����̣�ֱ�ӵ�����ǰ������������ε������뷽������ƽ��ֵ
                outputExpectation=cnn{i}.outputExpectationAvg{1};
                outputVarience=cnn{i}.outputVarianceAvg{1};
            end
            
            
            for bs=1:batchSize
                for fmi=1:featureMapNum
                    cnn{i}.preLayerOutputNorm{bs,fmi}=(preLayerOutput{bs,fmi}-outputExpectation(1,fmi))./sqrt(outputVarience(1,fmi)+epsilon);% ��׼��ǰһ�����
                    
                    cnn{i}.output{bs,fmi}=actFun(cnn{i}.preLayerOutputNorm{bs,fmi}*gamma(1,fmi)+beta(1,fmi));% ��ǰһ������������λ�Ʋ��ü��������
                end
            end
        end
        
        if (batchNum~=0)% batchNum~=0˵����ʱ��ѵ�����̣���Ҫ��¼��ǰ���ε������뷽��
            
            cnn{i}.outputExpectation{batchNumI,1}=outputExpectation;% �洢�ϲ������ֵ(����)
            cnn{i}.outputVarience{batchNumI,1}=outputVarience;% �洢�ϲ��������
            
            if(batchNumI==batchNum)
                outputExpectationAvg=zeros(size(cnn{i}.outputExpectation{1,1}));% �洢�ϲ��������
                outputVarienceAvg=zeros(size(cnn{i}.outputVarience{1,1}));% �洢�ϲ��������
                for bni=1:batchNum% �����������������뷽���ƽ��ֵ
                    outputExpectationAvg=outputExpectationAvg+cnn{i}.outputExpectation{bni,1};% �洢�ϲ������ֵ(����)
                    outputVarienceAvg=outputVarienceAvg+cnn{i}.outputVarience{bni,1};% �洢�ϲ��������
                    if(bni==batchNum)
                        outputExpectationAvg=outputExpectationAvg./batchNum;% ȡȫ����������ֵ�ľ�ֵ
                        outputVarienceAvg=(outputVarienceAvg./batchNum)*(batchSize/(batchSize-1));% ȡȫ���������������ƫ����
                    end
                end
                cnn{i}.outputExpectationAvg(1)={outputExpectationAvg};% ��¼����ѵ�����ε�����ƽ��ֵ
                cnn{i}.outputVarianceAvg(1)={outputVarienceAvg};% ��¼����ѵ�����εķ���ƽ��ֵ
            end
        end
    elseif strcmp(cnn{i}.layerType,'dropoutLayer')% ------------dropout��������------------
        
        preLayerOutput=cnn{i-1}.output;
        if(cnn{i}.dropoutRate>0)%���dropoutRate==0�򲻽���dropout
            for bs=1:batchSize% �Ա�batch�е�ÿһ������
                
                if (size(preLayerOutput{1,1},2)==1)&&(size(preLayerOutput{1,1},1)~=1)% �����һ����ȫ���Ӳ������ʽ����ȫ���Ӳ�ķ�ʽ����
                    if (batchNum~=0)% batchNum~=0˵����ʱ��ѵ�����̣���Ҫ�����ʶ��������dropout
                        cnn{i}.dropoutMask{bs,1}=random('uniform',0,1,size(preLayerOutput{1,1}))>cnn{i}.dropoutRate;% ���㵱ǰbatch������output��dropoutMask
                        cnn{i}.output{bs,1}=preLayerOutput{bs,1}.*cnn{i}.dropoutMask{bs,1};% �Ե�ǰ���������dropout
                    else% batchNum==0˵����ʱ��Ԥ����̣���Ҫ����������
                        cnn{i}.output{bs,1}=preLayerOutput{bs,1}*(1-cnn{i}.dropoutRate);
                    end
                    
                else% ��ȫ���Ӳ��ǰһ�����Ϊ��cell��װ������ӳ�����ΪbatchSize*featureMapNum
                    
                    featureMapNum=size(preLayerOutput,2);
                    
                    for fmi=1:featureMapNum
                        if (batchNum~=0)% batchNum~=0˵����ʱ��ѵ�����̣���Ҫ�����ʶ��������dropout
                            %                             cnn{i}.dropoutMask{bs,fmi}=random('uniform',0,1,size(preLayerOutput{1,1}))>cnn{i}.dropoutRate;% ���㵱ǰbatch������output��dropoutMask


                            if (cnn{i}.dropoutRate>1)
                                if(isnan(cnn{i}.dropoutMask{bs,fmi}(1))==1)
                                    cnn{i}.dropoutMask{bs,fmi}=ones(size(preLayerOutput{1,1}))*(rand>(cnn{i}.dropoutRate-floor(cnn{i}.dropoutRate)));% �ѵ�ǰ����ӳ��ͼ��������������Ӱ�dropoutRate��������
                                end
                            else
                                cnn{i}.dropoutMask{bs,fmi}=ones(size(preLayerOutput{1,1}))*(rand>cnn{i}.dropoutRate);% �ѵ�ǰ����ӳ��ͼ��������������Ӱ�dropoutRate��������
                            end
%                             cnn{i}.dropoutMask{bs,fmi}
                            
                            cnn{i}.output{bs,fmi}=preLayerOutput{bs,fmi}.*cnn{i}.dropoutMask{bs,fmi};% �Ե�ǰ���������dropout
                            
                        else% batchNum==0˵����ʱ��Ԥ����̣���Ҫ����������
                            cnn{i}.output{bs,fmi}=preLayerOutput{bs,fmi}.*(1-(cnn{i}.dropoutRate-floor(cnn{i}.dropoutRate)));
                        end
                    end
                end
                
            end
        else% ���������dropout����������ϲ����
            cnn{i}.output=preLayerOutput;
        end
        
        
    end
end

res=cnn{end}.output;
end