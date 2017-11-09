function [inputCell,output]=loadImgDataSet(pathStr,dataSetType,outputSize,imgSize,isDoGray,isDoBW,isDoColorReversal)
input={};
output=[];
outputAll=eye(outputSize);% Ŀ�����
loadProgress=waitbar(0,['��������',dataSetType,'����']);
for imgClassID=1:outputSize% �������ݼ�
    waitbar(imgClassID/outputSize,loadProgress,['����',dataSetType,'������ɣ�',num2str(roundn((imgClassID/outputSize)*100,-0)),'%']);
    %     for imgClassID=1:2% ������д�������ݼ�
    
    path=[pathStr,num2str(imgClassID)];% �������ݼ�·��
    
    %     for imgClassID=1:outputSize% ����ORL�������ݿ�
    %         path=[pathStr,'s',num2str(imgClassID)];% ����ORL�������ݿ�
    
    tempInput=readImgDir(path);
    tempOutput=repmat(outputAll(imgClassID,:),length(tempInput),1);
    input((length(input)+1):(length(input)+length(tempInput)))=tempInput;
    output=[output;tempOutput];
end
close(loadProgress);
input=input';
output=output*0.8+0.1;% ��Ŀ�����ȡֵ����Ϊ[0.1|0.9]


inputCell=imagePreprocess(input,dataSetType,imgSize,isDoGray,isDoBW,isDoColorReversal);% ����Ԥ����


end

