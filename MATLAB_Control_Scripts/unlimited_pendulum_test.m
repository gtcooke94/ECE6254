global G m l g umax
g = 10;
m = 1;
l = 1;
[X, L, G] = care([0 1; 0 0], [0; 1], diag([10,1]));
tspan = [0 15];
allcosts = [];
for iter = 1:100
    temp = rand(1,1);
    x01 = (rand(1,1)*pi/2+pi/2)*(-1*floor(temp)+1*ceil(temp));
    x02 = (rand(1,1)-0.5)*2;
    x0 = [x01; x02];
    %x0 = [-pi, 0];
    [tt,x] = ode45(@unlimited_pendulum, tspan, x0);
    U = [];
    cost = 0;
    for ii = 1:length(tt)
        v = -G*x(ii,:)';
        u = ((m*l^2)/3) * ((3*g/(2*l))*sin(x(ii,1)+pi) + v);
        U = [U u];
        if ii ~= length(tt)
            thiscost = ((mod(x(ii,1)+pi, 2*pi)-pi)^2+(0.1*x(ii,2))^2+0.001*u^2)*(tt(ii+1)-tt(ii));
        end
        cost = cost + thiscost;
    end
    Uvec = [Uvec; {U}];
    allcosts = [allcosts cost];
end
avgcost = mean(allcosts)
