function res=normalize(input)
%�����С��һ��������0-1
% res=(((input-repmat(min(input),size(input,1),1))./repmat((max(input)-min(input)),size(input,1),1))-0.5)*2;
res=((input-repmat(min(input),size(input,1),1))./repmat((max(input)-min(input)),size(input,1),1));

end

