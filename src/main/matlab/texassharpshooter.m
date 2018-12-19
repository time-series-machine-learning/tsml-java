function cd = texassharpshooter(name,x,y)
figure('Name', name);
%line(1,0);
%line(0,1);
scatter(x,y,'filled', 'MarkerFaceColor', [0 0 0]);
xlabel('Expected Accuracy Gain');
ylabel('Actual Accuracy Gain');

text(0.1, 0.9, {'False Positive'},'FontSize',14);
text(0.9, 0.9, {'True Positive'},'FontSize',14);
text(0.9, 0.1, {'False Negative'},'FontSize',14);
text(0.1, 0.1, {'True Negative'},'FontSize',14);


