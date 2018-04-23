function xdot = final_pendulum_prep_pos(t, x)
global G m l g umax

x1 = x(1);
x2 = x(2);

v = -G*(x - [pi; 0]);
u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1) + v);

if abs(u) >= umax
    u = umax*sign(u);
end

xdot = zeros(2,1);
xdot(1) = x2;
xdot(2) = -(3*g/(2*l)) * sin(x1+pi) + (3/(m*l^2))*u;
end