function xdot = pendulum_complete(t,x)
global G m l g umax x1_thresh x2_thresh tflag flag trigger

x1 = x(1);%mod(x(1)+pi, 2*pi)-pi;
x2 = x(2);

if flag == 1
    v = -G*(x + [pi; 0]);
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

    v = -G*(x - [pi; 0]);
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
        v = -G*x;
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
        v = -G*x;
        u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);
    end
elseif flag == 5
    u = 0;
end

if abs(u) >= umax
    u = umax*sign(u);
end

xdot = zeros(2,1);
xdot(1) = x2;
xdot(2) = -(3*g/(2*l)) * sin(x1+pi) + (3/(m*l^2))*u;
end
    