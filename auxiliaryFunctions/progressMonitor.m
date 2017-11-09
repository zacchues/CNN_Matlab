function progressMonitor(i,iMax,figureIndex,figTitle)


figure(figureIndex);
upper=0.7;
lower=0.3;
border=0.0;
borderTop=0.00;
if i==1
    clf
    hold on;
    title(figTitle)
%     xlabel('Åú´Î')
    plot([0,iMax],[upper,upper],'k--')
    plot([0,iMax],[lower,lower],'k--')
    for i2=0:iMax
        plot([i2,i2],[lower,upper],'k--')
    end
    axis([0,iMax,0,1]);
end
x=[i-1+border i-1+border i-border,i-border];
y=[lower+borderTop upper-borderTop upper-borderTop lower+borderTop];
patch(x,y,'b');
pause(0.0001)

end