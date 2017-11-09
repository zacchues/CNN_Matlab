function res=sigmoid(x,funType)

res=1./(1+exp(-x));

if nargin==2&&strcmp(funType,'derivative')
    res=res.*(1-res);
end
end