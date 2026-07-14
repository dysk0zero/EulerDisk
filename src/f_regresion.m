function [a, b, r, SEa, SEb] = f_regresion(x,y)
% Least-squares fit
% Model: y = b*x + a

% Force column vectors (important for Octave)
x = x(:);
y = y(:);

N = length(x);

mux = mean(x);
muy = mean(y);

SSxx = sum((x-mux).^2);
SSyy = sum((y-muy).^2);
SSxy = sum((x-mux).*(y-muy));

b = SSxy/SSxx;
a = muy - b*mux;

r = SSxy/sqrt(SSxx*SSyy);

s = sqrt((SSyy - b*SSxy)/(N-2));

SEb = s/sqrt(SSxx);
SEa = s*sqrt(1/N + mux^2/SSxx);

end