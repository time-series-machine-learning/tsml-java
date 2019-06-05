function [f] = roccurves(filepathandname, classifierNames,classValues,posClassProbs,posClassLabel,visible)
% roccurves, compute and draw ROC curve for 1 or more classifiers on a
% single set of test classifications. IT IS CURRENTLY ASSUMED THAT THE
% SCORES ARE FOR A TWO CLASS PROBLEM. Multiclass support may be added in
% future
%  
% if: 
%   n = number of instances
%   m = number of classifiers/models
%   C = number of classes/labels = 2  
%
% then: 
%   filepathandname : a string denoting the target filename and path for saving the
%        figure
%   classifierNames : a vector length m of strings (aka m by numcharacters char matrix) 
%        denoting the names of classifiers
%   classValues     : a vector length n of classvalues, denoting the class of
%        each instance. it is assumed that the posClassProbs provided are 
%        aligned for each classifier, this assumption may be relaxed in the future
%	posClassProbs   : an n by m matix of doubles/floats in the range 0 to 1 inclusive, 
%        such that each cell is the probability of the instance (row) being of 
%        the positive class as assigned by the classifier (column)
%   posClassLabel   : the class value that is to be considered the positive
%        class, which exists in the classValues vector. If no value is supplied,
%        the minority element in classValues is assumed to be the minority class
% 
% TEST DATA: 
%   n = 100
%   fname = 'roctest';
%   cnames = [ 'Classifier1'; 'Classifier2'; 'Classifier3'; 'Classifier4'; 'Classifier5' ];
%   classValues = (1:n)>50;
% 	posClassProbs = rand(size(cnames, 1), n);
%   roccurves(fname,cnames,classValues,posClassProbs)
% 
%
% based on perfcurve: https://uk.mathworks.com/help/stats/perfcurve.html

if nargin < 6 
   visible = 'off';
end

if nargin < 5
   posClassLabel = minmode(classValues);
end

primaryFontSize = 24;   %legend, axis labels
secondaryFontSize = 16; %axis tick labels

%f = figure('Name', filepathandname, 'visible', 'off');
f = figure('Name', filepathandname, 'visible', visible);

set(f,'Units','normalized');		
set(f,'Position',[0 0 0.7 0.5]);

m = size(classifierNames, 1);

legendLabels = cell(m,1);

for cid = 1:m
    classifierName = classifierNames(cid,:);
    posClassProbsC = posClassProbs(cid, :);
        
    % test/example cases 
    %classifierName = 'Random Guessing';
    %classValues = (1:100)>50;       % class values of each instance                   e.g [ 1 0 0 1 0 1 1... 0 ]
    %posClassProbsC = rand(1, 100);  % prob given for positive class of each instance, e.g [ 0.2, 0.5, 1, 0.5 ... 0.1 ]
    %posClassLabel = 1;              % positive class label                            e.g 1 

    [X,Y,T,AUC] = perfcurve(classValues,posClassProbsC,posClassLabel);

    if cid > 1
        hold on 
    end 
    
    plot(X,Y,'LineWidth',2)
    %title(sprintf('ROC for Classification by %s', classifierName))

    legendLabels{cid} = sprintf('%s, AUC=%.3f',classifierName,AUC);
end

set(gca,'FontSize',secondaryFontSize)

xlabel('False positive rate', 'FontSize',primaryFontSize) 
ylabel('True positive rate', 'FontSize',primaryFontSize)

if m > 1
    l = legend(legendLabels,'Location','Best');
    set(l,'FontSize',primaryFontSize);
end

orient(f,'landscape')
set(f,'PaperUnits','centimeters') 
set(f,'PaperPosition', [0 0 25 22]); %a4 page default

saveas(f,filepathandname); %fig file
print(f,filepathandname,'-dpdf'); %pdf file
end