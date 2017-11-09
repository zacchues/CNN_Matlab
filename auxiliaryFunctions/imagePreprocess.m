function inputCell=imagePreprocess(input,dataSetType,imgSize,isDoGray,isDoBW,isDoColorReversal)
% input�������С��ͼ��cell���ϣ�n��1��

prePrecessProgress=waitbar(0,['������ʼ',dataSetType,'����Ԥ����']);


% ======����������ת¼Ϊ��άcell��ʽ����Ԥ����======
if isDoGray==1 || isDoBW==1
    imgChNum=1;
else
    imgChNum=size(input{1},3);
end
inputCell=cell(size(input,1),imgChNum);



for inputID=1:size(input,1)% ͼƬ��
    % ======ͼƬԤ����======
    waitbar(inputID/size(input,1),prePrecessProgress,[dataSetType,'����Ԥ������ɣ�',num2str(roundn((inputID/size(input,1))*100,-0)),'%']);    
    tempInput=im2double(input{inputID});% ��ͼƬתΪdouble��ʾ
    
    %     tempInput=matrixNormalize(tempInput,[-1,1]);%������淶����[-1,1]����
    
    
    %     tempInput=tempInput((size(tempInput,1)/5):size(tempInput,1)*4/5,(size(tempInput,2)/5):size(tempInput,2)*4/5,:);% ֻȡ�м�3/5������
    
    tempInput=imresize(tempInput,[imgSize,imgSize]);% ����ͼƬ��СΪ[inputSize,inputSize]
    
    
    
    tempInput=matrixNormalize(tempInput,[-0.9,0.9]);%������淶����[-0.9,0.9]����
    
    if (isDoGray==1)
        tempInput=rgb2gray(tempInput);% �ҶȻ�
        if (isDoBW==1)% ��ֵ��
            tempInput=im2bw(tempInput,graythresh(tempInput));
        end
    end
    
    
    tempInput=matrixNormalize(tempInput,[-0.9,0.9]);%������淶����[-0.9,0.9]����
    
    if (isDoBW==1) && (isDoColorReversal==1)% �ڰ׷�ת
        tempInput=(-1)*tempInput;
    end
    
    %         tempInput=abs(tempInput-1);% �ڰ׷�ת���ڹ�һ����[0,1]�󷽿�ִ��
    
    % ======Ԥ�������
    
    for ch=1:imgChNum% ͨ����
        inputCell{inputID,ch}=tempInput(:,:,ch);% ת¼Ϊcell,�б�ʾͼƬ����ţ���ΪͼƬͨ����
    end
    
end
close(prePrecessProgress);
% ======Ԥ����ת¼����======



end