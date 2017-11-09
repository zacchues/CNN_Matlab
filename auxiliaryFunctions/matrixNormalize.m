function res=matrixNormalize(inputMatrix,scale)
% 将输入的矩阵inputMatrix中的所有元素规范化至scale=[low,high]范围内

maxItem=max(inputMatrix(:));
minItem=min(inputMatrix(:));

res=(((inputMatrix-minItem)./(maxItem-minItem)).*(scale(2)-scale(1)))+scale(1);


end