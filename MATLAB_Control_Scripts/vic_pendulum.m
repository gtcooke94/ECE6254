function xdot = vic_pendulum(t, x)
global G m l g umax flag

x1 = x(1);
x2 = x(2);
% x1temp = mod(x1+pi,2*pi)-pi;
% if t < 0.5
%     trigger = false;
%     ttrigger = 0;
% elseif abs(abs(x1temp)-pi) < 1 && abs(x(2)) < 1
%     trigger = true;
%     ttrigger = t;
% end
% if ~trigger
%     sprintf('here');
%     if x1temp > 0 && x(2) > 0
%         u = -2;
%     elseif x1temp < 0 && x(2) < 0
%         u = 2;
%     else
%         u = 0;
%     end
% else
    if t < 0.1
        flag = 1;
    elseif abs(abs(x1)-pi) < 0.3 
        if abs(x2) < 6 
            if x1 > 0
                flag = 1;
            else
                flag = 2;
            end

    %     v = -G*x;
    %     u = -((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);
    % 
    %     if abs(u) >= umax
    %         u = umax*sign(u);
    %     end
        elseif abs(x2) > 7
            if x1 > 0
                flag = 3;
            else
                flag = 4;
            end
        else
            flag = 5;
        end
    end
    
    if flag == 1
        u = -2;
    elseif flag == 2
        u = 2;
    elseif flag == 3
        u = 2;
    elseif flag == 4
        u = -2;
    else
        v = -G*x;
        u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x1+pi) + v);

        if abs(u) >= umax
            u = umax*sign(u);
        end
    end
        
% end

xdot = zeros(2,1);
xdot(1) = x2;
xdot(2) = -(3*g/(2*l)) * sin(x1+pi) + (3/(m*l^2))*u;
end