function [ f ] = timingsLinePlot( filename,evalSet )

%timings file is of form: 

%      label, classifier1, classifier2, ... 
%   dataset1,    c1d1time,    c2d1time, ...
%   dataset2,    c1d1time,    c2d1time, ...
%        ...,         ...,         ...

[timings, headers, ~] = xlsread([filename,'.csv']);
[ ~, inds ] = sort(median(timings, 2), 'ascend'); %mean
timings = timings(inds, :);

% data for x axis
dsets = headers(2:size(headers,1),1);
for i=1:length(dsets)
    dsets(i) = replace(dsets(i),'_','\_');
end
dsets = dsets(inds);
numDsets = length(dsets);
x = 1:numDsets;
numTicks = numDsets;
if numDsets > 10
    numTicks = 10;
end 
ticks = floor(linspace(1,numDsets,numTicks));
ticklabels = dsets(ticks);

%data for legend
classifiers = headers(1,2:size(headers,2));


% start drawing
f = figure('Name', filename, 'visible', 'off');

if (max(max(timings)) / min(min(timings)) > 100) %if timings span at least two orders of magnitude
    semilogy(x,timings, 'LineWidth',2); % plot on log scale
else 
    plot(x,timings, 'LineWidth',2); % else plot on regular scale 
end

set(gca,'FontSize',16)
xlabel(['Datasets ordered by average ',evalSet,' time, D=',int2str(length(dsets))], 'FontSize', 24);
ylabel(['Time, ',evalSet,' (ms)'], 'FontSize', 24);

xticks(ticks);
xtickangle(45);
xticklabels(ticklabels);

legend(classifiers, 'Location', 'northwest', 'FontSize', 24);

% save files
orient(f,'landscape')
set(f,'PaperUnits','normalized') 
set(f,'PaperPosition', [0.05 0 0.9 1]);
saveas(f,filename);
print(f,filename,'-dpdf'); %'-bestfit'

end

