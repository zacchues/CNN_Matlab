function CNN_Test(netSavePath,imgSize,isDoGray,isDoBW,isDoColorReversal)

cnn=load(netSavePath);

cnn=cnn.cnn;

[filename,pathname]=uigetfile({'*.bmp;*.jpg;*.tif;*.png;*.gif','All Image Files'},'请选择要预测的样本图片（可多选）','MultiSelect', 'on');

if iscell(filename)
    imgNum=size(filename,2);
else
    imgNum=size(filename,1);
end

testImgCell=cell(imgNum,1);
for i_img=1:imgNum
    if iscell(filename)
        testImgCell{i_img,1}=imread(strcat(pathname,filename{i_img}));
    else
        testImgCell{i_img,1}=imread(strcat(pathname,filename));
    end
end

testImg=imagePreprocess(testImgCell,'预测集',imgSize,isDoGray,isDoBW,isDoColorReversal);

for i_img=1:imgNum
    [tempOutput,~]=forwardPropagate_CNN(cnn,testImg(i_img,:),1,0);% 正向传播
    
    if iscell(filename)
        disp(['名为 ',filename{i_img},' 的样本图片对应的类别： ',num2str(find(tempOutput{1}==max(tempOutput{1})))]);
    else
        disp(['名为 ',filename,' 的样本图片对应的类别： ',num2str(find(tempOutput{1}==max(tempOutput{1})))]);
    end
end

end