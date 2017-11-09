function picgather=readImgDir(imgDir)

picstr=dir(imgDir);
[row,col]=size(picstr);
picgather=cell(row-2,1);
if(row>=3)
    for i=3:row
        picgather{i-2}=imread([imgDir,'/',picstr(i).name]);
    end
end
end