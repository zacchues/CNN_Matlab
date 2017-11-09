function inputCell=imagePreprocess(input,dataSetType,imgSize,isDoGray,isDoBW,isDoColorReversal)
% input是任意大小的图像cell集合，n行1列

prePrecessProgress=waitbar(0,['即将开始',dataSetType,'数据预处理']);


% ======将输入数据转录为二维cell格式，并预处理======
if isDoGray==1 || isDoBW==1
    imgChNum=1;
else
    imgChNum=size(input{1},3);
end
inputCell=cell(size(input,1),imgChNum);



for inputID=1:size(input,1)% 图片数
    % ======图片预处理======
    waitbar(inputID/size(input,1),prePrecessProgress,[dataSetType,'数据预处理完成：',num2str(roundn((inputID/size(input,1))*100,-0)),'%']);    
    tempInput=im2double(input{inputID});% 将图片转为double表示
    
    %     tempInput=matrixNormalize(tempInput,[-1,1]);%将矩阵规范化至[-1,1]区间
    
    
    %     tempInput=tempInput((size(tempInput,1)/5):size(tempInput,1)*4/5,(size(tempInput,2)/5):size(tempInput,2)*4/5,:);% 只取中间3/5的区域
    
    tempInput=imresize(tempInput,[imgSize,imgSize]);% 调整图片大小为[inputSize,inputSize]
    
    
    
    tempInput=matrixNormalize(tempInput,[-0.9,0.9]);%将矩阵规范化至[-0.9,0.9]区间
    
    if (isDoGray==1)
        tempInput=rgb2gray(tempInput);% 灰度化
        if (isDoBW==1)% 二值化
            tempInput=im2bw(tempInput,graythresh(tempInput));
        end
    end
    
    
    tempInput=matrixNormalize(tempInput,[-0.9,0.9]);%将矩阵规范化至[-0.9,0.9]区间
    
    if (isDoBW==1) && (isDoColorReversal==1)% 黑白翻转
        tempInput=(-1)*tempInput;
    end
    
    %         tempInput=abs(tempInput-1);% 黑白翻转，在归一化至[0,1]后方可执行
    
    % ======预处理结束
    
    for ch=1:imgChNum% 通道数
        inputCell{inputID,ch}=tempInput(:,:,ch);% 转录为cell,行表示图片数编号，列为图片通道数
    end
    
end
close(prePrecessProgress);
% ======预处理及转录结束======



end