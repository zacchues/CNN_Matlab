function res=matrixNormalize(inputMatrix,scale)
% ������ľ���inputMatrix�е�����Ԫ�ع淶����scale=[low,high]��Χ��

maxItem=max(inputMatrix(:));
minItem=min(inputMatrix(:));

res=(((inputMatrix-minItem)./(maxItem-minItem)).*(scale(2)-scale(1)))+scale(1);


end