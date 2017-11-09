function res=softmax_Loss(netOutput,obj,funType)

if nargin==3&&strcmp(funType,'loss')
    res=-sum(log(softmax_Loss(netOutput)),2);
elseif nargin==3&&strcmp(funType,'derivative')
    res=(softmax_Loss(netOutput)-obj);
else
    res=exp(netOutput)./repmat(sum(exp(netOutput),2),1,size(netOutput,2));
end

end