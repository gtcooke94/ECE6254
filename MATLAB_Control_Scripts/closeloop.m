% What if we closed the loop:
% x1_dot = x2
% x2_dot = (-3*g/(2*l)) * sin(x1+pi) + (3/(m*l^2))*u
clear 
clc
close all

global G m l g umax flag
% umax = 3.912402586;
umax = 2;
g = 10;
m = 1;
l = 1;
[X, L, G] = care([0 1; 0 0], [0; 1], diag([10,1]));

tspan = [0 8];
x0 = [pi; 0];
[tt,x] = ode45(@final_pendulum, tspan, x0);

V = [];
U = [];
for ii = 1:length(x)
    x1 = x(ii,1);
    x2 = x(ii,2);
    t = tt(ii);
    
    
    if 0 <= t && t < 0.8
        u = -2;
        v = -G*[x1; x2];
    elseif 0.8 <= t && t < 1.9
        u = 2;
        v = -G*[x1; x2];
    elseif 1.9 <= t && t < 2.32
        u = -1;
        v = -G*[x1; x2];
    elseif 2.32 <= t
        v = -G*[x1; x2];
        u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);

        if abs(u) >= umax
            u = umax*sign(u);
        end
    end
    
    V = [V v];
    U = [U u];
end

figure(1)
plot(tt,x)
title('State Space vs. t')
xlabel('time (s)')
ylabel('x_i(t)')
legend('x_1(t)', 'x_2(t)', 'Location', 'SouthOutside', 'Orientation', 'Horizontal')

figure(2)
plot(tt,U, tt,V)
title('Controls u(t) and v(t)')
xlabel('time (s)')
ylabel('u_c(t)')
legend('u(t)', 'v(t)', 'Location', 'SouthOutside', 'Orientation', 'Horizontal')

%% Animation
theta1 = x(:,1);

R = @(theta) [cos(theta) -sin(theta); sin(theta) cos(theta)];
G = @(x,y, theta) [R(theta), [x;y]; 0 0 1];
x1 = [];
y1 = [];
for ii = 1:length(theta1)
    T1 = theta1(ii);
    g01 = G(0,0, pi/2);
    g12 = G(0,0, T1);
    g23 = G(1,0, 0);
    q1 = g01*g12*g23*[0;0;1];
    x1(end+1) = q1(1);
    y1(end+1) = q1(2);
end

for jj = 1:length(x1)
    figure(2)
    plot([0 x1(jj)],[0 y1(jj)], 'r-', 'LineWidth', 3)
    axis([-1 1 -1 1])
    title('Plot of End Effector')
    xlabel('x')
    ylabel('y')
    hold off
    pause(1/300)
end

%% Big System:

clear 
clc
close all

global G m l g umax gu
umax = 3.912402586;
umax = 2;

gu = 2;
g = 0;
m = 1;
l = 1;
[X, L, G] = care([0 1; 0 0], [0; 1], diag([10,1]));

tspan = [0 10];
x0 = [pi; 0; pi; 0; 1];
[t,x] = ode45(@pendulum2, tspan, x0);

V = [];
U = [];
for ii = 1:length(x)
    v = -G*x(ii,1:2)';
    u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x(ii,1)+pi) + v);
    
    if abs(u) >= umax
        u = umax*sign(u);
    end
    
    V(end+1) = v;
    U(end+1) = u;
end

figure(1)
plot(t,x(:,1:4))
title('State Space vs. t')
xlabel('time (s)')
ylabel('x_i(t)')
legend('x_1(t)', 'x_2(t)', 'x_1_,_m', 'x_2_,_m', 'Location', 'SouthOutside', 'Orientation', 'Horizontal')

figure(2)
plot(t,U, t,V)
title('Controls u(t) and v(t)')
xlabel('time (s)')
ylabel('u_c(t)')
legend('u(t)', 'v(t)', 'Location', 'SouthOutside', 'Orientation', 'Horizontal')
figure(3)
plot(t, x(:,5))