function [a, b, r, SEa, SEb] = f_regresion(x,y)
% ajuste por minimos cuadrados
% modelo: y = b * x + a

N=length(x);
mux=mean(x); muy=mean(y);
SSxx=sum((x-mux).*(x-mux));
SSyy=sum((y-muy).*(y-muy));
SSxy=sum((x-mux).*(y-muy));
b=SSxy/SSxx;   % pendiente
a=muy-b*mux;   % ordenada origen
r=SSxy/sqrt(SSxx*SSyy); % coeficiente correlacion
s=sqrt((SSyy-b*SSxy)/(N-2));
SEb=s/sqrt(SSxx);   % error estandar
SEa=s*sqrt(1/N+mux*mux/SSxx);  % error estandar