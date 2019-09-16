function [data, headers] = readcsv(path, hasRowHeaders)
    fileID = fopen(path,'r');
    if hasRowHeaders
        headers = {};
        line = fgets(fileID);
        while line ~= -1
            line = split(line, ',');
            line = line(1);
            line = char(line);
            headers = cat(1, headers, line);
            line = fgets(fileID);
        end
    else
        headers = fgets(fileID); 
        headers = split(headers, ',');
        headers = headers';
        
        headers = strip(headers);
        %jamesl fixing trailing newline on last header
    end
    fclose(fileID);
    if hasRowHeaders
        data = csvread(path,1,1);
    else
        data = csvread(path,1);
    end
end

