function [cd,f] = buildDia(accsFile, a, pairwise, cliques)


%accs = csvread(accsFile);
%accs = accs';

[accs,labels,~] = xlsread(accsFile);
name = accsFile(1:(length(accsFile)-length('.csv')));

errs = 1 - accs;

clabels = cellstr(labels);

if pairwise == 1
    [cd,f] = criticaldifferencePairwise(name,errs,clabels,a,cliques);
else
    [cd,f] = criticaldifference(name,errs,clabels,a);
end 

%[~,name,~] = fileparts(accsFile);
%title(name);
%print(accsFile(1:(length(accsFile)-length('.csv'))), '-dpng');

orient(f,'landscape')
set(f,'Units','Inches');
set(f,'PaperUnits','normalized') 
set(f,'PaperPosition', [0.05 0 0.9 1]);
%pos = get(f,'Position');
%set('PaperSize',[pos(3)*1.3, pos(4)])
print(f,name,'-dpdf'); %'-bestfit'
%saveas(f,accsFile(1:(length(accsFile)-length('.csv'))), 'pdf');

end