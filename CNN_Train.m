function cnn=CNN_Train(CNN_StructPara,maxIterNum,costFunction,batchSize,isShowProgress,accuracy,netSavePath,pathStr,outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal,isLoadExistCNN,regularizationLamda,regularizationType)

[trainInput,trainOutput]=loadImgDataSet([pathStr,'train/'],'ѵ����',outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal);% ����ѵ����
[testInput,testOutput]=loadImgDataSet([pathStr,'test/'],'���Լ�',outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal);% ������Լ�


trainSetNum=size(trainOutput,1);% ������Լ�����
testSetNum=size(testOutput,1);% ������Լ�����
if batchSize>trainSetNum% ��Լһ���������
    batchSize=trainSetNum;
    disp(['����������batchSizeָ��Ϊ',num2str(batchSize),'����ѵ������������ֻ��',num2str(trainSetNum),',���Խ�batchSize�ĳ�batchSize=',num2str(trainSetNum)]);
elseif batchSize<1
    disp(['����batchSizeָ��Ϊ',num2str(batchSize),'С��1�ˣ����Խ�batchSize�ĳ�batchSize=1']);
    batchSize=1;
end

batchNum=floor(trainSetNum/batchSize);% ����ÿ��������
if min(size(trainInput{1,1},1),size(trainInput{1,1},2))==1
    dataDimension=1;% ����Ҫ�ȼ���һ�����������ά��
else
    dataDimension=2;% ����Ҫ�ȼ���һ�����������ά��
end

MSE_Test_Min=inf;
MSE_Train_Min=inf;

imgChNum=size(trainInput,2);% ͼƬͨ����
imgSize=size(trainInput{1,1},1);

trainInfo=['����ά��dataDimension=',num2str(dataDimension),',ͼƬͨ����imgChNum=',num2str(imgChNum),',ͼƬ�ߴ�Ϊ',num2str(imgSize),'x',num2str(imgSize),',ѵ��������trainSetNum=',num2str(trainSetNum),',���Լ�����testSetNum=',num2str(testSetNum),',���ߴ�batchSize=',num2str(batchSize),',ѵ������batchNum=',num2str(batchNum)];
disp(trainInfo);
if isLoadExistCNN==1
    cnn=load(netSavePath);
    cnn=cnn.cnn;
else
    cnn=buildCNN(CNN_StructPara,'image',imgSize,imgChNum,batchSize,batchNum);% ������άͼ�������
end
if isShowProgress==1
    wb=waitbar(0);
end

%% ����ѵ��
for i=1:maxIterNum
    
    % ������ǰ������ѵ�����Ͳ��Լ��ϵı���
    errTrainInfo=[];
%     errTestInfo=[];
    
    % ÿ�ε���ǰ��ѵ�����������ºϳ��µ�һ��batches
    sel=rand(1,trainSetNum);
    [~,n]=sort(sel);% ���ѡ����Ϊѵ������������
    
    batch=cell(batchNum,1);
    for b=1:batchNum% ÿ��������������
        lowI=batchSize*(b-1)+1;
        highI=batchSize*(b-1)+batchSize;
        batch{b}=n(lowI:highI);
    end
    
    if dataDimension==1% ѵ��һά����
        %         errAll1=0;
        %         errAll2=0;
        %         for si=1:trainSetNum
        %             [output,cnn]=forwardPropagate_CNN(cnn,(input_train(si,:))');
        %             err=(obj_train(si,:))'-output;
        %             cnn=backwardPropagate_CNN(cnn,err);
        %             errAll1=errAll1+(sum(err.^2));
        %         end
        %         MSE_Train(i)=errAll1./(trainSetNum);
        %
        %         for si=1:testSetNum
        %             [output,cnn1]=forwardPropagate_CNN(cnn,(input_Test(si,:))');
        %             err=(obj_Test(si,:))'-output;
        %             errAll2=errAll2+(sum(err.^2));
        %         end
        %         MSE_Test(i)=errAll2./(testSetNum);
        
    elseif dataDimension==2% ѵ����άͼ��
        %         errAll1=0;
        errAll2=0;
        
        for b=1:batchNum% ��ѵ����ѵ�����磬������������
            %             batchInfo=['����ѵ����',num2str(b),'������']
            [output,cnn]=forwardPropagate_CNN(cnn,trainInput(batch{b},:),b,batchNum);% ���򴫲�
            obj=trainOutput(batch{b},:);% ��ȡ��ǩ
            outputMatrix=nan(batchSize,size(output{1,1},1));
            for e=1:size(output,1)% ���תΪ������ʽ
                outputMatrix(e,:)=output{e,1}';
            end
            
            %                         outputMatrix
            errTrainInfo=[errTrainInfo,(outputMatrix-obj)'];
            
            %             for e=1:length(output)% ѵ����������
            %                 errTrain{e,1}=output{e,1}-obj(e,:)';
            %                 errTrainInfo=[errTrainInfo,errTrain{e,1}];
            %                 errAll1=errAll1+sum(errTrain{e,1}.^2);% �����ۼ�
            %             end
            %             sum(sum(errTrainInfo-errTrainInfo1))
            %             errTrainInfo
            %             pause
            
            
            
            
            
            %             ep=10^(-4);
            %             outputMatrix
            %             softmax_Loss(outputMatrix,obj,'derivative')
            %             sum(sum(softmax_Loss(outputMatrix,obj,'derivative')))
            %             sum(sum(((softmax_Loss(outputMatrix+ep,obj,'loss')-softmax_Loss(outputMatrix-ep,obj,'loss'))./(2*ep))))
            %             a='fds'
            %             pause
            
            lossMatrix=costFunction(outputMatrix,obj,'derivative');
            
            %             lossMatrix=exp(outputMatrix)./repmat(sum(exp(outputMatrix),2),1,size(outputMatrix,2))-obj;
            
            loss=cell(batchSize,1);
            for e=1:batchSize% ���תΪ����
                loss{e,1}=lossMatrix(e,:)';
            end
            
            cnn=backwardPropagate_CNN(cnn,loss,b,i,trainSetNum,regularizationLamda,regularizationType);% ���򴫲�
            
            if isShowProgress==1
                waitbar(b/batchNum,wb,['��',num2str(i),'��ѵ����ɣ�',num2str(roundn((b/batchNum)*100,-accuracy)),'%']);
            end
            
        end
        errAll1=sum(sum(errTrainInfo.^2));% �����ۼ�
        MSE_Train(i)=errAll1./(batchSize*batchNum);% ��¼��ǰѵ������MSEֵ
        
        classErr=zeros(1,size(testOutput,2));
        classNum=zeros(1,size(testOutput,2));
        for si=1:testSetNum% �ò��Լ���������Ԥ������
            [output,~]=forwardPropagate_CNN(cnn,testInput(si,:),1,0);% ��������������ݼ�����
            errTest=(testOutput(si,:))'-output{1,:};% ���Լ�������
            errAll2=errAll2+sum(errTest.^2);% �����ۼ�
            
            
            
            objClassIndex=find(testOutput(si,:)==max(testOutput(si,:)));
            classNum(objClassIndex)=classNum(objClassIndex)+1;
            
            if (abs(testOutput(si,objClassIndex)-output{1,:}(objClassIndex,:))<0.4)
                classErr(objClassIndex)=classErr(objClassIndex)+1;
            end
            
            
            if isShowProgress==1
                waitbar(si/testSetNum,wb,['��',num2str(i),'�ֲ�����ɣ�',num2str(roundn((si/testSetNum)*100,-accuracy)),'%'])
            end
            
%             errTestInfo=[errTestInfo,errTest];
        end
        MSE_Test(i)=errAll2./(size(testInput,1));% ��¼��ǰ���Լ���MSEֵ
        
    end
    
    
    CorrectlyTrainClassifiedSampleNum=sum(max(abs(errTrainInfo))<0.4);
%     CorrectlyTestClassifiedSampleNum=sum(max(abs(errTestInfo))<0.4);
    CorrectlyTestClassifiedSampleNum=sum(classErr);
    
    
    
    %% �����ǰ������ָ��
    
    
    %         %��ӡ����
    %         for printNetI=1:length(cnn)
    %             cnn{printNetI}
    %         end
    %         % pause
    
    if MSE_Train(i)<MSE_Train_Min
        MSE_Train_Min=MSE_Train(i);
        i_MSE_Train_Min=i;
    end
    if MSE_Test(i)<MSE_Test_Min
        MSE_Test_Min=MSE_Test(i);
        i_MSE_Test_Min=i;
        save(netSavePath,'cnn');% ֻ���ҵ����ž��������ʱ��洢����
    end
    
    if isShowProgress==1
        figure(1);
        clf;
        hold on;
        grid on;
        plot(MSE_Train,'b')
        plot(MSE_Test,'r')
        title(['ѵ����������ȷ�ʣ�',num2str((CorrectlyTrainClassifiedSampleNum/trainSetNum)*100),'% ,���Լ�������ȷ�ʣ�',num2str((CorrectlyTestClassifiedSampleNum/testSetNum)*100),'%']);
        legend('Train','Test');
        xlabel('epoch')
        ylabel('MSE')
        pause(0.001);
    end
    
    
    %             output1=cnn{1}.output{1}
    %
    %             deltaWeights2=cnn{2}.deltaWeights{1}
    %             weights2=cnn{2}.weights{1}
    %             output2=cnn{2}.output{1}
    %
    %             output3=cnn{3}.output{1}'
    %
    %             deltaWeights4=cnn{4}.deltaWeights{1}
    %         weights4=cnn{4}.weights{1}
    %             output3=cnn{3}.output{1}'
    %             output4=cnn{4}.output{1}'
    %             output5=cnn{5}.output{1}'
    %             err3=cnn{3}.err{1}'
    %             deltaWeights3=cnn{3}.deltaWeights{1}'
    %             err4=cnn{4}.err{1}'
    %             err5=cnn{5}.err{1}'
    %
    %         deltaWeights6=cnn{6}.deltaWeights{1}
    %         weights6=cnn{6}.weights{1}
    %         output6=cnn{6}.output
    %
    %         deltaWeights7=cnn{7}.deltaWeights{1}
    %         weights7=cnn{7}.weights{1}
    %         output7=cnn{7}.output{1}'
    %
    %         deltaWeights8=cnn{8}.deltaWeights{1}
    %         weights8=cnn{8}.weights{1}
    %         output8=cnn{8}.output{1}
    
    %         deltaWeights9=cnn{9}.deltaWeights{1}
    %         weights9=cnn{9}.weights{1}
    %         output9=cnn{9}.output{1}
    %         pause
    
    
    
    monitor=['����������',num2str(i),'  MSE_Train=',num2str(MSE_Train(i)),'  MSE_Test=',num2str(MSE_Test(i)),'  MSE_Train_Min=',num2str(MSE_Train_Min),'  MSE_Test_Min=',num2str(MSE_Test_Min),'  i_MSE_Train_Min=',num2str(i_MSE_Train_Min),'  i_MSE_Test_Min=',num2str(i_MSE_Test_Min)];
    trainMonitor=['ѵ��������������',num2str(trainSetNum),'  ����ȷ�����ѵ��������������',num2str(CorrectlyTrainClassifiedSampleNum),'  ѵ����������ȷ�ʣ�',num2str((CorrectlyTrainClassifiedSampleNum/trainSetNum)*100),'%'];
    testMonitor=['���Լ�����������',num2str(testSetNum),'  ����ȷ����Ĳ��Լ�����������',num2str(CorrectlyTestClassifiedSampleNum),'  ���Լ�������ȷ�ʣ�',num2str((CorrectlyTestClassifiedSampleNum/testSetNum)*100),'%'];
    testNumMonitor=['ÿ����������',num2str(classNum)];
    testErrorRateMonitor=['ÿ����ȷ�ʣ�',num2str(classErr./classNum)];
    
    disp(monitor);
    disp(trainMonitor);
    disp(testMonitor);
    disp(testNumMonitor);
    disp(testErrorRateMonitor);
    disp(' ');
    
    
    %     if ~(MSE_Test(i)>0.005)||~(MSE_Train(i)<50)
    %         break;
    %     end
    
    %     pause
    
    
end


end