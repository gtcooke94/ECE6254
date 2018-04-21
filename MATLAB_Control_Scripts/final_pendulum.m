function xdot = final_pendulum(t, x)
global G m l g umax

x1 = x(1);
x2 = x(2);

if 0 <= t && t < 0.8
    u = -2;
elseif 0.8 <= t && t < 1.9
    u = 2;

%     v = -G*x;
%     u = -((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);
% 
%     if abs(u) >= umax
%         u = umax*sign(u);
%     end
elseif 1.9 <= t && t < 2.32
    u = -1;
elseif 2.32 <= t
    v = -G*x;
    u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);

    if abs(u) >= umax
        u = umax*sign(u);
    end
end

xdot = zeros(2,1);
xdot(1) = x2;
xdot(2) = -(3*g/(2*l)) * sin(x1+pi) + (3/(m*l^2))*u;
end