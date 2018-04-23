% What if we closed the loop:
% x1_dot = x2
% x2_dot = (-3*g/(2*l)) * sin(x1+pi) + (3/(m*l^2))*u
clear 
clc
close all

global G m l g umax max_speed flag min_thresh max_thresh trigger theta_thresh ttrigger
global x1_thresh x2_thresh tflag
% umax = 3.912402586;
trigger = false;
umax = 2;
max_speed = 8;
min_thresh = 6;
max_thresh = 6.9;
theta_thresh = 0.025*pi;
flag = 0;
g = 10;
m = 1;
l = 1;
x1_thresh = 0.001*pi; x2_thresh = 0.01;
[X, L, G] = care([0 1; 0 0], [0; 1], diag([10,1]));

allcosts = [];
tspan = [0 15];
Uvec = [];
for iter = 1:100
    temp = rand(1,1);
    x01 = (rand(1,1)*pi/2+pi/2)*(-1*floor(temp)+1*ceil(temp));
    x02 = (rand(1,1)-0.5)*2;
    x0 = [x01; x02];
    %x0 = [-pi; 0]; % pi/6, -0.5 breaks
    if x0(1) == pi && x0(2) == 0
        flag = 3;
    elseif x0(1) == -pi && x0(2) == 0
        flag = 4;
    elseif x0(1) > 0
        flag = 2;
    elseif x0(1) < 0
        flag = 1;
    elseif all(x0) == 0
        flag = 5;
    end
    flagvec = flag;
    trigger = true;
    % min speed is 5.926
    % max speed is 6.975
    % x0 = [x01; x02];
    [tt,x] = ode45(@pendulum_complete, tspan, x0);
    finalflag = flag;
    flagvec = [flagvec finalflag];
    flag = flagvec(1);
    cost = 0;
    uvec = [];
    for ii = 1:length(x)
        t = tt(ii);
        if t > tflag
            flag = flagvec(2);
        end
        x1 = x(ii,1);%mod(x(1)+pi, 2*pi)-pi;
        x2 = x(ii,2);

        if flag == 1
            v = -G*(x(ii,:)' + [pi; 0]);
            u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1) + v);
            if (-abs(x1)+pi)<x1_thresh && abs(x2)<x2_thresh
                if x1 > 0
                    flag = 3;
                else
                    flag = 4;
                end
                tflag = t;
            end
        elseif flag == 2

            v = -G*(x(ii,:)' - [pi; 0]);
            u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1) + v);
            if (-abs(x1)+pi)<x1_thresh && abs(x2)<x2_thresh
                if x1 > 0
                    flag = 3;
                else
                    flag = 4;
                end
                tflag = t;
            end
        elseif flag == 3
            if trigger
                tflag = t;
                trigger = false;
            end
            ttemp = t - tflag;
            if ttemp < 0.8
                u = -2;
            elseif 0.8 <= ttemp && ttemp < 1.9
                u = 2;
            elseif 1.9 <= ttemp && ttemp < 2.32
                u = -1;
            elseif 2.32 <= ttemp
                v = -G*x(ii,:)';
                u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);
            end
        elseif flag == 4
            ttemp = t - tflag;
            if ttemp < 0.8
                u = 2;
            elseif 0.8 <= ttemp && ttemp < 1.9
                u = -2;
            elseif 1.9 <= ttemp && ttemp < 2.32
                u = 1;
            elseif 2.32 <= ttemp
                v = -G*x(ii,:)';
                u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);
            end
        elseif flag == 5
            u = 0;
        end

        if abs(u) >= umax
            u = umax*sign(u);
        end
        uvec = [uvec u];
        if ii ~= length(tt)
            thiscost = ((mod(x1+pi, 2*pi)-pi)^2+(0.1*x2)^2+0.001*u^2)*(tt(ii+1)-tt(ii));
        end
        cost = cost + thiscost;
    end
    Uvec = [Uvec; {uvec}];
    allcosts = [allcosts cost];
end
avgcost = mean(allcosts);
%[tt, x] = Pendulum_Euler(tspan, x0);

U = [];
for ind = 1:length(x)
    x1 = x(ind,1);
    x2 = x(ind,2);

    v = -G*(x(ind,:)' - [pi; 0]);
    u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1) + v);

    if abs(u) >= umax
        u = umax*sign(u);
    end
    U(end+1) = u;
end

figure(1)
plot(tt,x)
title(['State Space vs. t with x1_0=' num2str(x01) ' and x2_0=' num2str(x02)])
axis square
xlabel('time (s)')
ylabel('x_i(t)')
legend('x_1(t)', 'x_2(t)', 'Location', 'SouthOutside', 'Orientation', 'Horizontal')

figure(2)
plot(tt,Uvec{end})
title(['Control u(t) with x1_0=' num2str(x01) ' and x2_0=' num2str(x02)])
axis square
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
    figure(3)
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