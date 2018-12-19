function [cd,f] = criticaldifference(name,s,labels,alpha)
%
% CRITICALDIFFERNCE - plot a critical difference diagram
%
%    CRITICALDIFFERENCE(S,LABELS) produces a critical difference diagram [1]
%    displaying the statistical significance (or otherwise) of a matrix of
%    scores, S, achieved by a set of machine learning algorithms.  Here
%    LABELS is a cell array of strings giving the name of each algorithm.
%
%    References
%    
%    [1] Demsar, J., "Statistical comparisons of classifiers over multiple
%        datasets", Journal of Machine Learning Research, vol. 7, pp. 1-30,
%        2006.
%

%
% File        : criticaldifference.m
%
% Date        : Monday 14th April 2008
%
% Author      : Gavin C. Cawley
%
% Description : Sparse multinomial logistic regression using a Laplace prior.
%
% History     : 14/04/2008 - v1.00
%
% Copyright   : (c) Dr Gavin C. Cawley, April 2008.
%
%    This program is free software; you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation; either version 2 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program; if not, write to the Free Software
%    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
%


% Thanks to Gideon Dror for supplying the extended table of critical values.

if nargin < 3

   alpha = 0.1;

end

% convert scores into ranks

[N,k] = size(s);
[S,r] = sort(s');
idx   = k*repmat(0:N-1, k, 1)' + r';
R     = repmat(1:k, N, 1);
S     = S';

for i=1:N

   for j=1:k

      index    = S(i,j) == S(i,:);
      R(i,index) = mean(R(i,index));

   end

end

r(idx)  = R;
r       = r';

% compute critical difference

if alpha == 0.01

   qalpha = [0.000 2.576 2.913 3.113 3.255 3.364 3.452 3.526 3.590 3.646 ...
             3.696 3.741 3.781 3.818 3.853 3.884 3.914 3.941 3.967 3.992 ...
             4.015 4.037 4.057 4.077 4.096 4.114 4.132 4.148 4.164 4.179 ...
             4.194 4.208 4.222 4.236 4.249 4.261 4.273 4.285 4.296 4.307 ...
             4.318 4.329 4.339 4.349 4.359 4.368 4.378 4.387 4.395 4.404 ...
             4.412 4.420 4.428 4.435 4.442 4.449 4.456 ];

elseif alpha == 0.05

   qalpha = [0.000 1.960 2.344 2.569 2.728 2.850 2.948 3.031 3.102 3.164 ...
             3.219 3.268 3.313 3.354 3.391 3.426 3.458 3.489 3.517 3.544 ...
             3.569 3.593 3.616 3.637 3.658 3.678 3.696 3.714 3.732 3.749 ...
             3.765 3.780 3.795 3.810 3.824 3.837 3.850 3.863 3.876 3.888 ...
             3.899 3.911 3.922 3.933 3.943 3.954 3.964 3.973 3.983 3.992 ...
             4.001 4.009 4.017 4.025 4.032 4.040 4.046]; 

elseif alpha == 0.1

   qalpha = [0.000 1.645 2.052 2.291 2.460 2.589 2.693 2.780 2.855 2.920 ...
             2.978 3.030 3.077 3.120 3.159 3.196 3.230 3.261 3.291 3.319 ...
             3.346 3.371 3.394 3.417 3.439 3.459 3.479 3.498 3.516 3.533 ...
             3.550 3.567 3.582 3.597 3.612 3.626 3.640 3.653 3.666 3.679 ...
             3.691 3.703 3.714 3.726 3.737 3.747 3.758 3.768 3.778 3.788 ...
             3.797 3.806 3.814 3.823 3.831 3.838 3.846];

else

   error('alpha must be 0.01, 0.05 or 0.1');

end

cd = qalpha(k)*sqrt(k*(k+1)/(6*N));

f = figure('Name',name,'visible','off');

set(f,'Units','normalized');
set(f,'Position',[0 0 0.7 0.5]);

clf
axis off
axis([-0.5 1.5 0 140]);
axis xy 
tics = repmat((0:(k-1))/(k-1), 3, 1);
line(tics(:), repmat([100, 105, 100], 1, k), 'LineWidth', 2, 'Color', 'k');
tics = repmat(((0:(k-2))/(k-1)) + 0.5/(k-1), 3, 1);
line(tics(:), repmat([100, 102.5, 100], 1, k-1), 'LineWidth', 1, 'Color', 'k');
line([0 0 0 cd/(k-1) cd/(k-1) cd/(k-1)], [127 123 125 125 123 127], 'LineWidth', 1, 'Color', 'k');
h = text(0.5*cd/(k-1), 130, 'CD', 'FontSize', 20, 'HorizontalAlignment', 'center');

for i=1:k

   text((i-1)/(k-1), 110, num2str(k-i+1), 'FontSize', 18, 'HorizontalAlignment', 'center');

end

% compute average ranks

r       = mean(r);
[r,idx] = sort(r);

% compute statistically similar cliques

clique           = repmat(r,k,1) - repmat(r',1,k);
clique(clique<0) = realmax; 
clique           = clique < cd;

for i=k:-1:2

   if all(clique(i-1,clique(i,:))==clique(i,clique(i,:)))

      clique(i,:) = 0;

   end

end

n                = sum(clique,2);
clique           = clique(n>1,:);
n                = size(clique,1);

% labels displayed on the right

for i=1:ceil(k/2)

   line([(k-r(i))/(k-1) (k-r(i))/(k-1) 1.2], [100 100-5*(n+1)-10*i 100-5*(n+1)-10*i], 'Color', 'k');
   h = text(1.2, 100 - 5*(n+1)- 10*i + 5, num2str(r(i)), 'FontSize', 24, 'HorizontalAlignment', 'right');

   text(1.25, 100 - 5*(n+1) - 10*i + 4, labels{idx(i)}, 'FontSize', 28, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left');

end

% labels displayed on the left

for i=ceil(k/2)+1:k

   line([(k-r(i))/(k-1) (k-r(i))/(k-1) -0.2], [100 100-5*(n+1)-10*(k-i+1) 100-5*(n+1)-10*(k-i+1)], 'Color', 'k');

   text(-0.2, 100 - 5*(n+1) -10*(k-i+1)+5, num2str(r(i)), 'FontSize', 24, 'HorizontalAlignment', 'left');
   text(-0.25, 100 - 5*(n+1) -10*(k-i+1)+4, labels{idx(i)}, 'FontSize', 28, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right');
end

% group cliques of statistically similar classifiers

for i=1:size(clique,1)

   R = r(clique(i,:));

   line([((k-min(R))/(k-1)) + 0.015 ((k - max(R))/(k-1)) - 0.015], [100-5*i 100-5*i], 'LineWidth', 6, 'Color', 'k');

end

saveas(f, name);
% all done...

