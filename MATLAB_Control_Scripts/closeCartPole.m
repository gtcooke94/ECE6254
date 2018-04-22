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
tspan = [0 5];

x0 = (rand(4,1)-0.5)*0.1;
[t,x] = ode45(@cartpole2, tspan, x0);
%
far = find(abs(x(:,1))>1);
if isempty(far)
    far = 1;
end

figure(1)
plot(t,x(:,1:2))
title(['Plot of x_1(t) & x_2(t), hits | x_1| = 1 at t = ' num2str(t(far(1)))])
xlabel('time (s)')
ylabel('x_i(t)')
legend('x_1(t)', 'x_2(t)')

figure(2)
plot(t,x(:,3:4))
xlabel('time (s)')
ylabel('x_i(t)')
legend('x_3(t)', 'x_4(t)')
U = [];
for qq = 1:length(t)
    x1 = x(qq,1);
    x2 = x(qq,2);
    x3 = x(qq,3);
    x4 = x(qq,4);
    
    if x2>0 && x3>0
        u = F;
    elseif x2>0 && x3<0
        u = -F;
    elseif x2<0 && x3>0
        u = F;
    elseif x2<0 && x3<0
        u = -F;
    else
        if abs(x3) > 6*pi/360
            u = -u;
        end
    end
    U(end+1) = u;
end

figure(3)
plot(t,U)
xlabel('time (s)')
ylabel('u(t)')
legend('u(t)')
% Animation
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
    figure(4)
    plot([x1(jj) x2(jj)],[0 y2(jj)], 'r-', 'LineWidth', 3)
    axis([-2.4 2.4 0 2.4])
    axis square
    title('Plot of End Effector')
    xlabel('x')
    ylabel('y')
    hold off
    pause(1/300)
    if abs(x1(jj)) > 2.4 || abs(theta1(jj)) > 24*pi/360
        tfinal = t(jj);
        title(['Plot of End Effector ended at t = ' num2str(tfinal)])
        break;
    end
end