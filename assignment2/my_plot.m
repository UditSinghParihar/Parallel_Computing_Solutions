x1 = [1, 2, 4, 6];
y1 = [4.71, 2.42, 1.22, 0.83];

x2 = [1, 2,6,10,15,20];
y2 = [16.4,8.3, 2.8, 3.1, 2.25, 1.8];

h = figure
plot(x2, y2)
pause(10)
savefig(h, 'save.fig')