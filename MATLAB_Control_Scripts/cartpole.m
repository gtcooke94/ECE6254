function xdot = cartpole(t,x)
global g mc mp l F c

x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);
u = F;
% if abs(x1) > .25 && abs(x3) < 12*pi/360
%     u = -sign(x1) * F;    
% elseif abs(x3) > 6*pi/360
%     u = -u;
% end

if abs(x3) > 6*pi/360
    u = -u;
end

input = (u + mp * l * x4^2 * sin(x3)) / (mp+mc);

xdot = zeros(4,1);

xdot(3) = x4;
xdot(4) = (g*sin(x3) - cos(x3)*input) / (l*(4/3 - mp*cos(x3)^2/(mp+mc)));
xdot(1) = x2;
xdot(2) = input - mp*l*xdot(4)*cos(x3)/(mp+mc);
end