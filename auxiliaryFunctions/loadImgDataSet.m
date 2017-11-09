function [inputCell,output]=loadImgDataSet(pathStr,dataSetType,outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal)
input={};
output=[];
outputAll=eye(outputSize);% 目标输出
loadProgress=waitbar(0,['即将载入',dataSetType,'数据']);
for imgClassID=1:outputSize% 载入数据集
    waitbar(imgClassID/outputSize,loadProgress,['载入',dataSetType,'数据完成：',num2str(roundn((imgClassID/outputSize)*100,-0)),'%']);
    %     for imgClassID=1:2% 载入手写数字数据集
    
    path=[pathStr,num2str(imgClassID)];% 构建数据集路径
    
    %     for imgClassID=1:outputSize% 载入ORL人脸数据库
    %         path=[pathStr,'s',num2str(imgClassID)];% 载入ORL人脸数据库
    
    tempInput=readImgDir(path);
    tempOutput=repmat(outputAll(imgClassID,:),length(tempInput),1);
    input((length(input)+1):(length(input)+length(tempInput)))=tempInput;
    output=[output;tempOutput];
end
close(loadProgress);
input=input';
output=output*0.8+0.1;% 将目标输出取值调整为[0.1|0.9]


inputCell=imagePreprocess(input,dataSetType,imgSize,isDoGray,isDoBW,isDoColorReversal);% 数据预处理


end

