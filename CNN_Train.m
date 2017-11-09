function cnn=CNN_Train(CNN_StructPara,maxIterNum,costFunction,batchSize,isShowProgress,accuracy,netSavePath,pathStr,outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal,isLoadExistCNN,regularizationLamda,regularizationType)

[trainInput,trainOutput]=loadImgDataSet([pathStr,'train/'],'训练集',outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal);% 载入训练集
[testInput,testOutput]=loadImgDataSet([pathStr,'test/'],'测试集',outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal);% 载入测试集


trainSetNum=size(trainOutput,1);% 计算测试集数量
testSetNum=size(testOutput,1);% 计算测试集数量
if batchSize>trainSetNum% 规约一下输入参数
    batchSize=trainSetNum;
    disp(['批量样本数batchSize指定为',num2str(batchSize),'，但训练集样本容量只有',num2str(trainSetNum),',所以将batchSize改成batchSize=',num2str(trainSetNum)]);
elseif batchSize<1
    disp(['批量batchSize指定为',num2str(batchSize),'小于1了，所以将batchSize改成batchSize=1']);
    batchSize=1;
end

batchNum=floor(trainSetNum/batchSize);% 计算每批样本量
if min(size(trainInput{1,1},1),size(trainInput{1,1},2))==1
    dataDimension=1;% 这里要先计算一下输入的数据维度
else
    dataDimension=2;% 这里要先计算一下输入的数据维度
end

MSE_Test_Min=inf;
MSE_Train_Min=inf;

imgChNum=size(trainInput,2);% 图片通道数
imgSize=size(trainInput{1,1},1);

trainInfo=['数据维度dataDimension=',num2str(dataDimension),',图片通道数imgChNum=',num2str(imgChNum),',图片尺寸为',num2str(imgSize),'x',num2str(imgSize),',训练集数量trainSetNum=',num2str(trainSetNum),',测试集数量testSetNum=',num2str(testSetNum),',批尺寸batchSize=',num2str(batchSize),',训练批次batchNum=',num2str(batchNum)];
disp(trainInfo);
if isLoadExistCNN==1
    cnn=load(netSavePath);
    cnn=cnn.cnn;
else
    cnn=buildCNN(CNN_StructPara,'image',imgSize,imgChNum,batchSize,batchNum);% 创建二维图像的网络
end
if isShowProgress==1
    wb=waitbar(0);
end

%% 迭代训练
for i=1:maxIterNum
    
    % 评估当前网络在训练集和测试集上的表现
    errTrainInfo=[];
%     errTestInfo=[];
    
    % 每次迭代前将训练集打乱重新合成新的一批batches
    sel=rand(1,trainSetNum);
    [~,n]=sort(sel);% 随机选择作为训练集的样本号
    
    batch=cell(batchNum,1);
    for b=1:batchNum% 每批样本轮流计算
        lowI=batchSize*(b-1)+1;
        highI=batchSize*(b-1)+batchSize;
        batch{b}=n(lowI:highI);
    end
    
    if dataDimension==1% 训练一维向量
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
        
    elseif dataDimension==2% 训练二维图像
        %         errAll1=0;
        errAll2=0;
        
        for b=1:batchNum% 用训练集训练网络，分批带入样本
            %             batchInfo=['正在训练第',num2str(b),'批样本']
            [output,cnn]=forwardPropagate_CNN(cnn,trainInput(batch{b},:),b,batchNum);% 正向传播
            obj=trainOutput(batch{b},:);% 获取标签
            outputMatrix=nan(batchSize,size(output{1,1},1));
            for e=1:size(output,1)% 输出转为矩阵形式
                outputMatrix(e,:)=output{e,1}';
            end
            
            %                         outputMatrix
            errTrainInfo=[errTrainInfo,(outputMatrix-obj)'];
            
            %             for e=1:length(output)% 训练集误差计算
            %                 errTrain{e,1}=output{e,1}-obj(e,:)';
            %                 errTrainInfo=[errTrainInfo,errTrain{e,1}];
            %                 errAll1=errAll1+sum(errTrain{e,1}.^2);% 误差和累加
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
            for e=1:batchSize% 输出转为矩阵
                loss{e,1}=lossMatrix(e,:)';
            end
            
            cnn=backwardPropagate_CNN(cnn,loss,b,i,trainSetNum,regularizationLamda,regularizationType);% 反向传播
            
            if isShowProgress==1
                waitbar(b/batchNum,wb,['第',num2str(i),'轮训练完成：',num2str(roundn((b/batchNum)*100,-accuracy)),'%']);
            end
            
        end
        errAll1=sum(sum(errTrainInfo.^2));% 误差和累加
        MSE_Train(i)=errAll1./(batchSize*batchNum);% 记录当前训练集的MSE值
        
        classErr=zeros(1,size(testOutput,2));
        classNum=zeros(1,size(testOutput,2));
        for si=1:testSetNum% 用测试集测试网络预测能力
            [output,~]=forwardPropagate_CNN(cnn,testInput(si,:),1,0);% 单条输入测试数据计算结果
            errTest=(testOutput(si,:))'-output{1,:};% 测试集误差计算
            errAll2=errAll2+sum(errTest.^2);% 误差和累加
            
            
            
            objClassIndex=find(testOutput(si,:)==max(testOutput(si,:)));
            classNum(objClassIndex)=classNum(objClassIndex)+1;
            
            if (abs(testOutput(si,objClassIndex)-output{1,:}(objClassIndex,:))<0.4)
                classErr(objClassIndex)=classErr(objClassIndex)+1;
            end
            
            
            if isShowProgress==1
                waitbar(si/testSetNum,wb,['第',num2str(i),'轮测试完成：',num2str(roundn((si/testSetNum)*100,-accuracy)),'%'])
            end
            
%             errTestInfo=[errTestInfo,errTest];
        end
        MSE_Test(i)=errAll2./(size(testInput,1));% 记录当前测试集的MSE值
        
    end
    
    
    CorrectlyTrainClassifiedSampleNum=sum(max(abs(errTrainInfo))<0.4);
%     CorrectlyTestClassifiedSampleNum=sum(max(abs(errTestInfo))<0.4);
    CorrectlyTestClassifiedSampleNum=sum(classErr);
    
    
    
    %% 输出当前的性能指标
    
    
    %         %打印网络
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
        save(netSavePath,'cnn');% 只在找到更优精度网络的时候存储网络
    end
    
    if isShowProgress==1
        figure(1);
        clf;
        hold on;
        grid on;
        plot(MSE_Train,'b')
        plot(MSE_Test,'r')
        title(['训练集分类正确率：',num2str((CorrectlyTrainClassifiedSampleNum/trainSetNum)*100),'% ,测试集分类正确率：',num2str((CorrectlyTestClassifiedSampleNum/testSetNum)*100),'%']);
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
    
    
    
    monitor=['迭代次数：',num2str(i),'  MSE_Train=',num2str(MSE_Train(i)),'  MSE_Test=',num2str(MSE_Test(i)),'  MSE_Train_Min=',num2str(MSE_Train_Min),'  MSE_Test_Min=',num2str(MSE_Test_Min),'  i_MSE_Train_Min=',num2str(i_MSE_Train_Min),'  i_MSE_Test_Min=',num2str(i_MSE_Test_Min)];
    trainMonitor=['训练集样本数量：',num2str(trainSetNum),'  被正确分类的训练集样本数量：',num2str(CorrectlyTrainClassifiedSampleNum),'  训练集分类正确率：',num2str((CorrectlyTrainClassifiedSampleNum/trainSetNum)*100),'%'];
    testMonitor=['测试集样本数量：',num2str(testSetNum),'  被正确分类的测试集样本数量：',num2str(CorrectlyTestClassifiedSampleNum),'  测试集分类正确率：',num2str((CorrectlyTestClassifiedSampleNum/testSetNum)*100),'%'];
    testNumMonitor=['每类样本数：',num2str(classNum)];
    testErrorRateMonitor=['每类正确率：',num2str(classErr./classNum)];
    
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