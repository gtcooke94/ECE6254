%% Run Cartpole
clear
clc
close all

global g mc mp l F c u
c = 0;
g = 9.8;
mc = 1;
mp = 0.1;
l = 0.5;
F = 10;
u = F;
tspan = [0 8];
x0 = [0;0;0;.0];
[t,x] = ode45(@cartpole, tspan, x0);
far = find(abs(x(:,1))>1);
figure(1)
plot(t,x(:,1:2))
title(['Plot of x_1(t) & x_2(t), hits |x_1| = 1 at t = ' num2str(t(far(1)))])
xlabel('time (s)')
ylabel('x_i(t)')
legend('x_1(t)', 'x_2(t)')

% figure(2)
% plot(t,x(:,3:4))
% xlabel('time (s)')
% ylabel('x_i(t)')
% legend('x_3(t)', 'x_4(t)')

%% Animation
theta1 = x(:,3);
pos = x(:,1);
R = @(theta) [cos(theta) -sin(theta); sin(theta) cos(theta)];
G = @(x,y, theta) [R(theta), [x;y]; 0 0 1];
x1 = [];
y1 = 0;
x2 = [];
y2 = [];
for ii = 1:length(theta1)
    T1 = theta1(ii);
    xpos = pos(ii);
    g01 = G(xpos,0, pi/2);
    g12 = G(0,0, T1);
    g23 = G(l,0, 0);
    q1 = g01*g12*g23*[0;0;1];
    x1(end+1) = xpos;
    x2(end+1) = q1(1);
    y2(end+1) = q1(2);
end

for jj = 1:length(x2)
    figure(3)
    plot([x1(jj) x2(jj)],[0 y2(jj)], 'r-', 'LineWidth', 3)
    axis([-2.4 2.4 0 2.4])
    axis square
    title('Plot of End Effector')
    xlabel('x')
    ylabel('y')
    hold off
    pause(1/300)
    if abs(x1(jj)) > 2.4 || abs(theta1(jj)) > 24*pi/360
        break;
    end
end