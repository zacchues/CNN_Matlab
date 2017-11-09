function res=MSE(netOutput,obj,funType)


if strcmp(funType,'loss')
    res=sum(sum((netOutput-obj).^2))./2;
elseif strcmp(funType,'derivative')
    res=netOutput-obj;
end


end