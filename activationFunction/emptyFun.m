function res=emptyFun(x,funType)

res=x;
if nargin==2&&strcmp(funType,'derivative')
    res=1;
end
end