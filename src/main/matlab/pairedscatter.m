function f = pairedscatter(name,x,y,labels,statistic,descriptor)

try
    % Some code that may generate an error.
    f = figure('Name', name, 'visible', 'off');
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end


try
    % Some code that may generate an error.
    scatter(x,y,'filled', 'MarkerFaceColor', [0 0 0]);
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end


try
    % Some code that may generate an error.
    set(gca,'FontSize',16);
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end


try
    % Some code that may generate an error.
    xlabel([labels{1}, ' ', statistic], 'FontSize', 20);
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end


try
    % Some code that may generate an error.
    ylabel([labels{2}, ' ', statistic], 'FontSize', 20);
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end


try
    % Some code that may generate an error.
    limit = ceil(max(max(x), max(y)));
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end


try
    % Some code that may generate an error.
    x1 = [0 0 limit];
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end


try
    % Some code that may generate an error.
    y1 = [0 limit limit];
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end


try
    % Some code that may generate an error.
    patch(x1,y1,'cyan', 'FaceAlpha', 0.1, 'EdgeAlpha', 0);
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end


try
    % Some code that may generate an error.
    text(0.1*limit, 0.8*limit, {labels{2},sprintf('is %s here', descriptor)},'FontSize',24);
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end



try
    % Some code that may generate an error.
    text(0.7*limit, 0.2*limit, {labels{1},sprintf('is %s here', descriptor)},'FontSize',24);
catch ME
    % Some error occurred if you get here.
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end

orient(f,'landscape');
set(f,'PaperUnits','normalized') ;
set(f,'PaperPosition', [0.05 0 0.9 1]);

set(f, 'CreateFcn', 'set(gcbo,''Visible'',''on'')');

saveas(f,name);
print(f,name,'-dpdf'); %'-bestfit'

end