function res=rectifier(x,funType)

res=max(0,x);
if nargin==2&&strcmp(funType,'derivative')
    res=x>0;
end


end

