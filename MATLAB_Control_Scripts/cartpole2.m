function xdot = cartpole2(t,x)
global g mc mp l F

x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

if x2>0 && x3>0
    u = F;
elseif x2>=0 && x3<0
    u = -F;
elseif x2<0 && x3>0
    u = F;
elseif x2<=0 && x3<0
    u = -F;
else
    if abs(x3) > 6*pi/360
        u = -u;
    end
end

input = (u + mp * l * x4^2 * sin(x3)) / (mp+mc);

xdot = zeros(4,1);

xdot(3) = x4;
xdot(4) = (g*sin(x3) - cos(x3)*input) / (l*(4/3 - mp*cos(x3)^2/(mp+mc)));
xdot(1) = x2;
xdot(2) = input - mp*l*xdot(4)*cos(x3)/(mp+mc);
end