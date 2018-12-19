function buildDiasInDirectory(folder, pairwise, pval)

if nargin < 3
   pval = 0.05;
end

k = dir(strcat(folder, '*.csv'));
filenames = {k.name}';

if pairwise == 0
    for i=1:length(filenames)
        buildDia(char(strcat(folder, filenames(i))), pval, 0);
    end
else %pairwise == 1
    for i=1:length(filenames)
        accsFile = char(filenames(i));
        name = accsFile(1:(length(accsFile)-length('.csv')));
        expr = fileread(strcat(folder, name, '_cliques.txt'));
        eval(expr);
        buildDia(char(strcat(folder, filenames(i))), pval, 1, cliques);
    end
end

end