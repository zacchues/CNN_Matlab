function res=tanhFun(x,funType)


res=(exp(x)-exp(-x))./(exp(x)+exp(-x));

if nargin==2&&strcmp(funType,'derivative')
    res=1-res.^2;
end


end