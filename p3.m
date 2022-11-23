
% read table of legible file type
Tab = readtable("Nerlove1963.txt", 'ReadVariableNames', 0);
%assign variables
Cost = table2array(Tab(:,"Var1"));
logTC = log(Cost);
Output =table2array(Tab(:,"Var2"));
logQ = log(Output);
logQ2 = logQ' * logQ;
Plabor = table2array(Tab(:,"Var3"));
logPL = log( Plabor);
Pcapital = table2array(Tab(:,"Var4"));
logPK = log( Pcapital);
Pfuel = table2array(Tab(:,"Var5"));
logPF = log( Pfuel);
n=145;
minb7 = 4.143135;
maxb7 = 8.644882;
nitt = 1000;
bhatc = zeros(5, nitt);

%part c

% create non linear model
Y = logTC-logPF;
X = ones(n,5);
X(:,2) = logQ;
X(:,3) = logPL - logPF;
X(:,4) = logPK - logPF;

%1000 B7 values
B7 = linspace(minb7,maxb7,nitt);

%calculate squared residuals for each B7
for i=1:nitt
    z = logQ.*(1+exp(-logQ + B7(1,i))).^(-1);
    X(:,5) = z;
    X2 = X'*X;
    XY= X'*Y;
    bhatc(:,i)=inv(X2)*(XY);
    r = Y-X*bhatc(:,i);
    r2(i) = r'*r;
    
end

%NLLS is maximized where resid^2 is minimized
[val,row]=min(r2);
display('Beta hat 1:')
bhatc(1,row)
display('Beta hat 2:')
bhatc(2,row)
display('Beta hat 3:')
bhatc(3,row)
display('Beta hat 4:')
bhatc(4,row)
display('Beta hat 5:')
1-bhatc(3,row)-bhatc(4,row)
display('Beta hat 6:')
bhatc(5,row)
display('Beta hat 7:')
B7(1,row)

%part d
%using out calculated B7
zi = logQ.*(1+exp(-logQ + B7(1,row))).^(-1);
%beta matrix for part d
bhatd = [bhatc(1,row), bhatc(2,row), bhatc(3,row), bhatc(4,row), 1-bhatc(3,row)-bhatc(4,row), bhatc(5,row), B7(1,row)];
X(:,5) = zi;

%residuals
rd = Y(:,1)-(X(:,1)*bhatd(1)+X(:,2)*bhatd(2)+X(:,3)*bhatd(3)+X(:,4)*bhatd(4)+X(:,5)*bhatd(6));
rd2 = rd.*rd;
rd2sum = sum(rd2);

% g(zi, theta 0)
g = zi.^2.*(exp(bhatd(7)-logQ(:)))./logQ;

%derivative w. r. t. Beta
for i=1:n
    d(:,i) = [X(i,:)';-bhatd(6)*(Output(i)*exp(bhatd(7))*logQ(i))/(Output(i)+exp(bhatd(7)))^2];

end

%derivative squared
d2 = d*d';

%E[x x']
h = d2/n;
sig2 = rd2sum/(n-6)

display('The var cov matrix is')
Var = 1/n * sig2 * inv(h)

display('the standard error of B1 is:')
sqrt(Var(1,1))
display('the standard error of B2 is:')
sqrt(Var(2,2))
display('the standard error of B3 is:')
sqrt(Var(3,3))
display('the standard error of B4 is:')
sqrt(Var(4,4))
display('the standard error of B6 is:')
sqrt(Var(5,5))
display('the standard error of B7 is:')
sqrt(Var(6,6))
