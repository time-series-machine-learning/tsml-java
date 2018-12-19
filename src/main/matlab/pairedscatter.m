function f = pairedscatter(name,x,y,labels,statistic)
f = figure('Name', name, 'visible', 'off');

scatter(x,y,'filled', 'MarkerFaceColor', [0 0 0]);
xlabel([labels{1}, ' ', statistic], 'FontSize', 16);
ylabel([labels{2}, ' ', statistic], 'FontSize', 16);

if strcmp(statistic, 'NLL')
    textlabel = ' worse';
else
    textlabel = ' better';
end

limit = ceil(max(max(x), max(y)));

x1 = [0 0 limit];
y1 = [0 limit limit];
patch(x1,y1,'cyan', 'FaceAlpha', 0.1, 'EdgeAlpha', 0);
text(0.1*limit, 0.8*limit, {labels{2},strcat('is',textlabel,' here')},'FontSize',20);
text(0.7*limit, 0.2*limit, {labels{1},strcat('is',textlabel,' here')},'FontSize',20);
saveas(f, name);

orient(f,'landscape')
set(f,'PaperUnits','normalized') 
set(f,'PaperPosition', [0.05 0 0.9 1]);
print(f,name,'-dpdf'); %'-bestfit'

end