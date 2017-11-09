function res=softplus(x,funType)
%SOFTPLUS Summary of this function goes here
%   Detailed explanation goes here

res=max(0,log(1+exp(x)));

if nargin==2&&strcmp(funType,'derivative')
    res=1./(1+exp(-x));
end



end

