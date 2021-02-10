function f = pairedscatter(filename, x_scores, y_scores, classifier_labels, statistic_label, difference_descriptor, fit_line, origin_is_zero)

if (nargin < 7)
    fit_line = false;
end 
if (nargin < 8)
    origin_is_zero = true;
end

f = figure('Name', filename, 'visible', 'off');

scatter(x_scores,y_scores,'filled', 'MarkerFaceColor', [0 0 0]);

set(gca,'FontSize',16);

xlabel([classifier_labels{1}, ' ', statistic_label], 'FontSize', 20);
ylabel([classifier_labels{2}, ' ', statistic_label], 'FontSize', 20);

if (origin_is_zero)
    min_limit = min(floor(min(min(x_scores), min(y_scores))), 0);
else
    min_limit = floor(min(min(x_scores), min(y_scores)));
end
max_limit = ceil(max(max(x_scores), max(y_scores)));
xlim([min_limit max_limit]); % axis limits
ylim([min_limit max_limit]);

x1 = [min_limit min_limit max_limit];
y1 = [min_limit max_limit max_limit];
patch(x1,y1,[0.2,0.9,0.9], 'FaceAlpha', 0.3, 'EdgeAlpha', 0);
%patch(x1,y1,'cyan', 'FaceAlpha', 0.1, 'EdgeAlpha', 0);

t1_x = min_limit + 0.1*(max_limit - min_limit);
t1_y = min_limit + 0.8*(max_limit - min_limit);
t2_x = min_limit + 0.7*(max_limit - min_limit);
t2_y = min_limit + 0.2*(max_limit - min_limit);

text(t1_x, t1_y, {classifier_labels{2},sprintf('is %s here', difference_descriptor)},'FontSize',24);
text(t2_x, t2_y, {classifier_labels{1},sprintf('is %s here', difference_descriptor)},'FontSize',24);


if (fit_line == true) 
    hold on
    linearCoefficients = polyfit(x_scores, y_scores, 1);
    xFit = linspace(min_limit, max_limit);
    yFit = polyval(linearCoefficients, xFit);
    
    plot(xFit, yFit, 'LineWidth', 2)
    
    lbl = sprintf('y = %.2fx + %.2f', linearCoefficients(1), linearCoefficients(2));

%     t3_x = min_limit + 0.2*(max_limit - min_limit);
%     t3_y = min_limit + 0.1*(max_limit - min_limit);
%     text(t3_x, t3_y, lbl, 'FontSize',24);
end


orient(f,'landscape');
set(f,'PaperUnits','normalized') ;
set(f,'PaperPosition', [0.05 0 0.9 1]);

set(f, 'CreateFcn', 'set(gcbo,''Visible'',''on'')');

saveas(f,filename);
print(f,filename,'-dpdf'); %'-bestfit'

end