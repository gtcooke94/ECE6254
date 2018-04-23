function xdot = unlimited_pendulum(t, x)
global G m l g umax

x1 = x(1);
x2 = x(2);

v = -G*x;
u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);


xdot = zeros(2,1);
xdot(1) = x2;
xdot(2) = -(3*g/(2*l)) * sin(x1+pi) + (3/(m*l^2))*u;
end
