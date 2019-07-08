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
%	posClassProbs   : an m by n matix of doubles/floats in the range 0 to 1 inclusive, 
%        such that each element is the probability of the instance (column) being of 
%        the positive class as assigned by the classifier (row)
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
   visible = 'on';
end

if nargin < 5
   posClassLabel = minmode(classValues);
end

primaryFontSize = 24;   %legend, axis labels
secondaryFontSize = 16; %axis tick labels

%f = figure('Name', filepathandname, 'visible', 'off');
f = figure('Name', filepathandname, 'visible', 'off');

set(f,'Units','normalized');		
set(f,'Position',[0 0 0.7 0.5]);

m = size(classifierNames, 1);
n = length(classValues);

%X = zeros(n+1,m);
%Y = zeros(n+1,m);

X = cell(m,1);
Y = cell(m,1);

AUC = zeros(m,1);
legendLabels = cell(m,1);

for cid = 1:m        
    [x,y,~,auc] = perfcurve(classValues,posClassProbs(cid, :),posClassLabel);
    
    AUC(cid) = auc;
    X{cid} = x;
    Y{cid} = y;
    legendLabels{cid} = sprintf('%s, AUC=%.3f',classifierNames(cid,:),auc);
end
    
if m > 1
    [~, inds] = sort(AUC, 'descend');
    X = X(inds);
    Y = Y(inds);
    legendLabels = legendLabels(inds);
end

plot(X{1}, Y{1}, 'LineWidth', 2);

if m > 1
    hold on
    for i=2:m
       plot(X{i}, Y{i}, 'LineWidth', 2);
    end 
end 

%plot(X,Y,'LineWidth',2)
%title(sprintf('ROC for Classification by %s', classifierName))

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

set(f, 'CreateFcn', 'set(gcbo,''Visible'',''on'')');

saveas(f,filepathandname); %fig file
print(f,filepathandname,'-dpdf'); %pdf file
end