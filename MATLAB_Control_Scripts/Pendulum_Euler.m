function [tt,x] = Pendulum_Euler(tspan, x0)

global G m l g umax max_speed min_thresh max_thresh flag theta_thresh
dt = 0.01;
tt = tspan(1):dt:tspan(2);
x = x0';
flag = 2;
counter = 10;
trigger = false;
for ind = 1:length(tt)-1

x1 = x(ind,1);
x2 = x(ind,2);

if counter >= 10 && (trigger && abs(flag) ~= 2)
    if abs(abs(x1)-pi) <= theta_thresh && 0<=x2 && x2<=min_thresh
        % going to the right (Not fast enough yet)
        flag = 1;
    elseif abs(abs(x1)-pi) <= theta_thresh && -min_thresh<=x2 && x2<=0
        % going to the left (Not fast enough yet)
        flag = -1;
    elseif abs(abs(x1)-pi) <= theta_thresh && min_thresh<x2 && x2 < max_thresh
        % going to the right (Correct Speed!)
        flag = -2;
    elseif abs(abs(x1)-pi) <= theta_thresh && -max_thresh<x2 && x2 < -min_thresh
        % going to the left (Correct Speed!)
        flag = 2;
    elseif abs(abs(x1)-pi) <= theta_thresh && max_thresh <= x2
        % going to the right (too fast!)
        flag = -3;
    elseif abs(abs(x1)-pi) <= theta_thresh && x2 <= -max_thresh
        % going to the left (too fast!)
        flag = 3;
    elseif abs(x2) < 0.6 && abs(flag) ~= 2
        flag = -flag;
    end
    counter = 0;
    trigger = true;
end
counter = counter + 1;

switch flag
    case 1
        u = 2;
    case 2
        v = -G*x(ind,:)';
        u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);
    case 3
        u = 1.5;
    case -1
        u = -2;
    case -2
        v = -G*(x(ind,:)'-[2*pi; 0]);
        u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);        
    case -3
        u = -1.5;
    otherwise
        u = 0;
end
        

if abs(u) >= umax
    u = umax*sign(u);
end

xdot = zeros(2,1);
xdot(1) = x2;
if abs(x2) >= max_speed
    xdot(1) = max_speed*sign(x2);
end
xdot(2) = -(3*g/(2*l)) * sin(x1+pi) + (3/(m*l^2))*u;

newx = [0 0];
newx(1) = x1+dt*xdot(1);
newx(2) = x2+dt*xdot(2);
if abs(newx(2)) >= max_speed
    newx(2) = max_speed*sign(newx(2));
end

x = [x; newx];
end

end