function f = pairedscatter(name,x,y,labels,statistic,descriptor)
f = figure('Name', name, 'visible', 'off');

scatter(x,y,'filled', 'MarkerFaceColor', [0 0 0]);
set(gca,'FontSize',16)
xlabel([labels{1}, ' ', statistic], 'FontSize', 20);
ylabel([labels{2}, ' ', statistic], 'FontSize', 20);

%if strcmp(statistic, 'NLL')
%    descriptor = ' worse';
%else
%    descriptor = ' better';
%end
%descriptor = strcat(' ', descriptor);

limit = ceil(max(max(x), max(y)));

x1 = [0 0 limit];
y1 = [0 limit limit];
patch(x1,y1,'cyan', 'FaceAlpha', 0.1, 'EdgeAlpha', 0);
text(0.1*limit, 0.8*limit, {labels{2},sprintf('is %s here', descriptor)},'FontSize',24);
text(0.7*limit, 0.2*limit, {labels{1},sprintf('is %s here', descriptor)},'FontSize',24);

orient(f,'landscape')
set(f,'PaperUnits','normalized') 
set(f,'PaperPosition', [0.05 0 0.9 1]);

saveas(f,name);
print(f,name,'-dpdf'); %'-bestfit'

end