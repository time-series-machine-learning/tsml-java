[accs, accheaders, ~] = xlsread('CAWPEminusBestIndividualOnTrainAccs.csv');
[sigs, sigheaders, ~] = xlsread('CAWPEminusBestIndividualOnTrainAccs_sigDiffs.csv');

accs = accs';
medians = median(accs);

[medians origInds] = sort(medians);

sortedaccs = zeros(size(accs));
sortedsigs = zeros(size(sigs));
for i = 1 : length(accs)
   sortedaccs(:,i) = accs(:,origInds(i)); 
   sortedsigs(i) = sigs(origInds(i));
end


%boxplot(sortedaccs,'PlotStyle','compact')
%boxplot(sortedaccs)
boxplot(sortedaccs,    'Symbol','o',    'OutlierSize',4,     'Colors',[.1 .1 .1],         'MedianStyle','target'  ) %  
xlabel('Random-component-set Ensemble IDs, ordered by increasing median difference between CAWPE and best component of each set')
ylabel('Diff in acc between ensemble and best individual, 121 UCI datasets')

set(gca,'xtick',[ 0, 50, 100, 150, 200 ])
set(gca,'xticklabel',{'0', '50', '100', '150', '200'}, 'fontsize',24)


boxcolors = [];
colors = ['r', 'b'];
for i = 1 : length(sortedsigs)
    if sortedsigs(i)
        boxcolors(i) = colors(1);
    else 
        boxcolors(i) = colors(2);
    end 
end 

%findobj returns them last to first. 
boxcolors = fliplr(boxcolors);

h = findobj(gca,'Tag','Box');
for j=1:length(h)
   patch(get(h(j),'XData'),get(h(j),'YData'),boxcolors(j),'FaceAlpha',.5);
end

c = get(gca, 'Children');
leg = legend(c(191:192), 'Significant difference', 'No Significant difference');
leg.FontSize = 24;
leg.Location = 'northwest';

zeroline = refline(0, 0);
zeroline.Color = 'r';
zeroline.LineWidth = 2;