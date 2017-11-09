function cnn=buildCNN(cnnStructure,inputType,inputSize,imgChNum,batchSize,batchNum)

cnn=cell(size(cnnStructure,1),1);
for i=1:length(cnnStructure)% ���ݽṹ��������Ӧ��ʼ��ÿһ������
    
    if strcmp(cnnStructure{i}{1},'inputLayer')% ------------������ʼ��------------
        tempLayer.layerType=cnnStructure{i}{1};% ��ʼ���ò�����
%         tempLayer.err={};% �����ԭʼ�����ǲв����ֱ����������Ȩֵ������ͨ�����򴫲�����в
%         tempLayer.learningRate=0;% �ò�ѧϰ��
        tempLayer.weights={};
        tempLayer.bias={};
        tempLayer.outputFMSize=inputSize;% �������������ӳ��ͼ�ĳߴ�
        
        if strcmp(inputType,'image')% �������������ͼƬ
            for bs=1:batchSize% batch�ж�ÿ��������batch sample�ֱ��ʼ��
                for fmn=1:imgChNum% ��ʼ������������ƫ��ֵ���вÿ������ӳ����һ�������ʾ
                    tempLayer.output{bs,fmn}=nan(tempLayer.outputFMSize);% ��ʼ������ͼ�εĴ洢�ռ�
                end
            end
        elseif strcmp(inputType,'vector')% �������������һά����
            for bs=1:batchSize% batch�ж�ÿ��������batch sample�ֱ��ʼ��
                tempLayer.output{bs}=nan(tempLayer.outputFMSize,1);% ��ʼ��һά�����Ĵ洢�ռ�
            end
        end
        
        cnn(i)={tempLayer};% ���һ��
        
    elseif strcmp(cnnStructure{i}{1},'convLayer')% ------------������ʼ��------------
        tempLayer.layerType=cnnStructure{i}{1};% �ò�����
        tempLayer.actFun=cnnStructure{i}{2};% �ò㼤���
        tempLayer.learningRate=cnnStructure{i}{3};% �ò�ѧϰ��
        tempLayer.momentum=cnnStructure{i}{4};% �ò����
        initWeightRange=cnnStructure{i}{5};% �ò�Ȩ�س�ʼ��ʱ�������Χ
        tempLayer.maxWeightRange=cnnStructure{i}{6};% �ò�Ȩ�ص����ɱ䷶Χ
        tempLayer.kernelSize=cnnStructure{i}{7};% �ò����˳ߴ�
        tempLayer.connType=cnnStructure{i}{8};% �ò���������
        
        tempLayer.outputFMSize=size(cnn{i-1}.output{1,1},1)-tempLayer.kernelSize+1;% ͨ���ϲ��������ߴ���㱾������ӳ��ߴ�
        
        if strcmp(tempLayer.connType,'full')% �������ӷ�ʽȷ����Ҫ������ӳ������
            tempLayer.featureMapNum=cnnStructure{i}{9};% �þ��������ӳ������
            for fmn=1:tempLayer.featureMapNum% ��ʼ������������ƫ��ֵ���вÿ������ӳ����һ�������ʾ
                tempLayer.bias{fmn}=zeros(tempLayer.outputFMSize);
                tempLayer.deltaBias{fmn}=zeros(tempLayer.outputFMSize);% ��ʼ����ֵ�ı���
                for bs=1:batchSize% batch�ж�ÿ��������batch sample�ֱ��ʼ��
                    tempLayer.output{bs,fmn}=nan(tempLayer.outputFMSize);
                    tempLayer.err{bs,fmn}=nan(tempLayer.outputFMSize);
                end
            end
            
            preLayerOutputNum=size(cnn{i-1}.output,2);% ��ȡ�ϲ����������ӳ��ͼ����
            
            tempLayer.weights=cell(preLayerOutputNum,tempLayer.featureMapNum);% ����ǰһ�������������������ӳ������������˳ߴ�����ʼ������Ȩֵ
            % cell���к�Ϊ��һ������ӳ��ţ��к�Ϊ��������ӳ���
            for i1=1:preLayerOutputNum
                for i2=1:tempLayer.featureMapNum
                    % weights���кŴ����ϲ��Ӧ������ӳ��㣬�кŴ������Ӧ������ӳ��㣬������������������м���
                    tempLayer.weights{i1,i2}=random('uniform',-initWeightRange,initWeightRange,tempLayer.kernelSize,tempLayer.kernelSize);% �����ʼ��Ȩֵ
                    tempLayer.deltaWeights{i1,i2}=zeros(tempLayer.kernelSize,tempLayer.kernelSize);% ��ʼ��Ȩֵ�ı���
                end
            end
        elseif strcmp(tempLayer.connType,'local')% �������Ӿ�����������ӳ�������ӵľ���˵�
            tempLayer.connIndexMatrix=cnnStructure{i}{9};% ��ȡ���Ӿ���
            preLayerOutputNum=size(cnn{i-1}.output,2);
            if size(tempLayer.connIndexMatrix,2)~=preLayerOutputNum% ������������Ӿ����ǰ�����������Ƿ���ǰһ������ӳ��ͼ������ͬ���粻ͬ�����������Ϣ
                inf1='output size cannot match,please check your input Matrix.';
                inf2=' Current layer:';
                preLayerFeatureMapNum=' PreLayer feature map num:';
                currentLayerFeatureMapNum=' Current feature map num:';
                disp([inf1,inf2,num2str(i),preLayerFeatureMapNum,num2str(size(tempLayer.connIndexMatrix,2)),currentLayerFeatureMapNum,num2str(length(cnn{i-1}.output))]);
                pause;
            end
            
            tempLayer.featureMapNum=size(tempLayer.connIndexMatrix,1);% ���Ӿ����е��б�ʾǰһ������ӳ��ͼ���б�ʾ��������ӳ��ͼ
            
            for fmn=1:tempLayer.featureMapNum% ��ʼ������������ƫ��ֵ���вÿ������ӳ����һ�������ʾ
                tempLayer.bias{fmn}=zeros(tempLayer.outputFMSize);
                tempLayer.deltaBias{fmn}=zeros(tempLayer.outputFMSize);% ��ʼ����ֵ�ı���
                for bs=1:batchSize% batch�ж�ÿ��������batch sample�ֱ��ʼ��
                    tempLayer.output{bs,fmn}=nan(tempLayer.outputFMSize);
                    tempLayer.err{bs,fmn}=nan(tempLayer.outputFMSize);
                end
            end
            
            tempLayer.weights=cell(preLayerOutputNum,tempLayer.featureMapNum);% ����ǰһ�������������������ӳ������������˳ߴ�����ʼ������Ȩֵ
            
            for i1=1:preLayerOutputNum% cell���к�Ϊ��һ������ӳ��ţ��к�Ϊ��������ӳ���
                for i2=1:tempLayer.featureMapNum
                    % weights���кŴ����ϲ��Ӧ������ӳ��㣬�кŴ������Ӧ������ӳ��㣬������������������м���
                    if (tempLayer.connIndexMatrix(i2,i1))~=0
                        tempLayer.weights{i1,i2}=random('uniform',-initWeightRange,initWeightRange,tempLayer.kernelSize,tempLayer.kernelSize);% �����ʼ��Ȩֵ
                        tempLayer.deltaWeights{i1,i2}=zeros(tempLayer.kernelSize,tempLayer.kernelSize);% ��ʼ��Ȩֵ�ı���
                    else
                        tempLayer.weights{i1,i2}=nan;
                    end
                end
            end
        end
        cnn(i)={tempLayer};
        
    elseif strcmp(cnnStructure{i}{1},'subSampLayer')% ------------���������ʼ��------------
        tempLayer.layerType=cnnStructure{i}{1};% ��ʼ���ò�����
        tempLayer.actFun=cnnStructure{i}{2};
        tempLayer.learningRate=cnnStructure{i}{3};
        tempLayer.momentum=cnnStructure{i}{4};
        initWeightRange=cnnStructure{i}{5};% �ò�Ȩ�س�ʼ��ʱ�������Χ
        tempLayer.maxWeightRange=cnnStructure{i}{6};% �ò�Ȩ�ص����ɱ䷶Χ
        tempLayer.size=cnnStructure{i}{7};
        tempLayer.isDirConn=cnnStructure{i}{8};
        tempLayer.sampType=cnnStructure{i}{9};
        tempLayer.outputFMSize=ceil(size(cnn{i-1}.output{1,1},1)/tempLayer.size);% ͨ���ϲ��������ߴ�������ֱ��ʼ��㱾������ӳ�����ߴ�
        
        
        
        for fmn=1:size(cnn{i-1}.output,2)% ��ʼ���������������ƫ��ֵ���в�
            tempLayer.bias{fmn}=zeros(tempLayer.outputFMSize);
            for bs=1:batchSize% batch�ж�ÿ��������batch sample�ֱ��ʼ��
                tempLayer.output{bs,fmn}=nan(tempLayer.outputFMSize);
                tempLayer.err{bs,fmn}=nan(tempLayer.outputFMSize);
            end
        end
        tempLayer.weights={};% ֱ������Ҫ����Ȩֵ����Ĭ��ȫΪ1
        tempLayer.deltaWeights=0;
        tempLayer.deltaBias=0;
        
        cnn(i)={tempLayer};
        
    elseif strcmp(cnnStructure{i}{1},'fullConnLayer')% ------------ȫ���Ӳ��ʼ��------------
        tempLayer.layerType=cnnStructure{i}{1};% ��ʼ���ò�����
        tempLayer.actFun=cnnStructure{i}{2};
        tempLayer.learningRate=cnnStructure{i}{3};
        tempLayer.momentum=cnnStructure{i}{4};
        initWeightRange=cnnStructure{i}{5};% �ò�Ȩ�س�ʼ��ʱ�������Χ
        tempLayer.maxWeightRange=cnnStructure{i}{6};% �ò�Ȩ�ص����ɱ䷶Χ
        outputSize=cnnStructure{i}{7};
        
        tempLayer.bias{1}=zeros(outputSize,1);
        tempLayer.deltaBias{1}=zeros(outputSize,1);% ��ʼ����ֵ�ı���
        for bs=1:batchSize% batch�ж�ÿ��������batch sample�ֱ��ʼ��
            tempLayer.bias{bs}=zeros(outputSize,1);
            tempLayer.deltaBias{bs}=zeros(outputSize,1);% ��ʼ����ֵ�ı���
            tempLayer.output{bs,1}=nan(outputSize,1);% ��ʼ��ȫ���Ӳ������ƫ��ֵ���в�
            tempLayer.err{bs,1}=nan(outputSize,1);
        end
        
        % ��ɺ����һ�����
        preLayerOutput=cnn{i-1}.output;% ��һ�������һά����������󼯣�
        
        %         preLayerOutput1={};% ��ɺ������
        %         preOutputSize=cnn{i-1}.outputFMSize;% �ϲ�����ӳ��ĳߴ�
        preLayerFeatureMapNum=size(preLayerOutput,2);% �ϲ�����ӳ�����������ֻ��һ��������(ȫ���Ӳ�)���ֵΪ1
        if max(size(preLayerOutput))~=1||min(size(preLayerOutput{1}))~=1% �������Ϊ�������ӳ���
            %
            %             for bs=1:batchSize% batch�ж�ÿ��������batch sample�ֱ��ʼ��
            %                 tempOutput=[];
            %                 for i1=1:preLayerFeatureMapNum% ��ɺ��������һ����������ӳ���������һ�в���������
            %                     for i2=1:preOutputSize
            %                         tempOutput=[tempOutput,preLayerOutput{bs,i1}(i2,:)];
            %                     end
            %                 end
            %                 preLayerOutput1{bs}=tempOutput;
            %             end
            preLayerOutputLength=size(preLayerOutput{bs,1},1)*size(preLayerOutput{bs,1},2)*preLayerFeatureMapNum;
        else% �������Ϊ����������
            %             for bs=1:batchSize% batch�ж�ÿ��������batch sample�ֱ��ʼ��
            %                 preLayerOutput1{bs}=preLayerOutput{bs,1};
            %             end
            preLayerOutputLength=size(preLayerOutput{1},1);
        end
        tempLayer.weights={random('uniform',-initWeightRange,initWeightRange,preLayerOutputLength,outputSize)};% ÿ�д����ϲ��һ������Ӧ�������н���Ȩֵ
        tempLayer.deltaWeights={zeros(preLayerOutputLength,outputSize)};% ��ʼ��Ȩֵ�ı���
        cnn(i)={tempLayer};
        
    elseif strcmp(cnnStructure{i}{1},'batchNormalizationLayer')% ------------batch normalization���ʼ��------------
        tempLayer.layerType=cnnStructure{i}{1};% ��ʼ���ò�����
        tempLayer.actFun=cnnStructure{i}{2};
        tempLayer.output=cnn{i-1}.output;% ��ȡ�ϲ�output
        tempLayer.err=cnn{i-1}.output;% ��ȡ�ϲ�output
        tempLayer.outputExpectation=cell(batchNum,1);% ��¼ÿһ�����������
        tempLayer.outputVariance=cell(batchNum,1);% ��¼ÿһ�����������
        tempLayer.preLayerOutputNorm=cnn{i-1}.output;% �洢ǰһ������ı�׼������
        
        if (size(tempLayer.output{1,1},2)==1)&&(size(tempLayer.output{1,1},1)~=1)% ����output�ж�ǰһ���ǲ���ȫ���Ӳ�
%             tempLayer.beta=normrnd(0,1,1,size(tempLayer.output{1,1},1));% ��0Ϊ���ĵ���̬�ֲ���ʼ��
%             tempLayer.gamma=normrnd(1,1,1,size(tempLayer.output{1,1},1));% ��1Ϊ���ĵ���̬�ֲ���ʼ��
            tempLayer.beta=zeros(1,size(tempLayer.output{1,1},1));% ȫ0��ʼ��
            tempLayer.gamma=ones(1,size(tempLayer.output{1,1},1));% ȫ1��ʼ��
            %             tempLayer.beta=zeros(1,size(tempLayer.output{1,1},1));
            %             tempLayer.gamma=ones(1,size(tempLayer.output{1,1},1));
            tempLayer.outputExpectationAvg=cell(1,1);% ��¼����ѵ�����ε�����ƽ��ֵ
            tempLayer.outputVarianceAvg=cell(1,1);% ��¼����ѵ�����εķ���ƽ��ֵ
        else% �������ȫ���Ӳ�
%             tempLayer.beta=normrnd(0,1,1,size(tempLayer.output,2));% ��0Ϊ���ĵ���̬�ֲ���ʼ��
%             tempLayer.gamma=normrnd(1,1,1,size(tempLayer.output,2));% ��1Ϊ���ĵ���̬�ֲ���ʼ��
            tempLayer.beta=zeros(1,size(tempLayer.output,2));% ȫ0��ʼ��
            tempLayer.gamma=ones(1,size(tempLayer.output,2));% ȫ1��ʼ��
            tempLayer.outputExpectationAvg=cell(1,size(tempLayer.output,2));% ��¼����ѵ�����ε�����ƽ��ֵ
            tempLayer.outputVarianceAvg=cell(1,size(tempLayer.output,2));% ��¼����ѵ�����εķ���ƽ��ֵ
        end
        tempLayer.learningRate=cnnStructure{i}{3};
        tempLayer.momentum=cnnStructure{i}{4};
        
        cnn(i)={tempLayer};
        
    elseif strcmp(cnnStructure{i}{1},'dropoutLayer')% ------------dropout���ʼ��------------
        
        tempLayer.layerType=cnnStructure{i}{1};% ��ʼ���ò�����
        tempLayer.dropoutRate=cnnStructure{i}{2};
        tempLayer.output=cnn{i-1}.output;% ��ȡ�ϲ�output����ʼ������output
        tempLayer.err=cnn{i-1}.output;% ��ȡ�ϲ�err����ʼ������output
        tempLayer.dropoutMask=cnn{i-1}.output;% ��ʼ��dropoutMask����С���������������ͬ����������ĸ�������Ӧ�ñ�Ϊ0
        
        cnn(i)={tempLayer};
    end
    
    clear tempLayer;
end

end