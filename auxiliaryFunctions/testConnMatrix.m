function testConnMatrix(maxFMN,minCN)


connIndexMatrix=[];
if minCN<=maxFMN
    for i=minCN:maxFMN
        loopNum=nchoosek(1:maxFMN,i)
        tempConnIndexMatrix=zeros(size(loopNum,1),maxFMN);
        for j=1:size(loopNum,1)
            tempConnIndexMatrix(j,loopNum(j,:))=1;
        end
        connIndexMatrix=[connIndexMatrix;tempConnIndexMatrix];
    end
    
    
end
connIndexMatrix=connIndexMatrix'

end