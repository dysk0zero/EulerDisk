% analysis_EulerDisk.m
% Matlab / Octave script for analysis of audio file corresponding to
% rotatio of Euler Disk

clear;
clc;
close all;

% GNU Octave packages
pkg load signal


filename = 'data/wav/recording_04.wav';
[x,fs]=audioread(filename);
if length(x) < 3e6
    x(3e6)=0;                 % extend signal to 30 seconds
end
x=x+randn(size(x))*1e-3;  % signal contaminated with AGWN (because of the extension with 0 values)
N=length(x);
t=(0:(N-1))'/fs;

%% Seal en el tiempo
figure(1);
plot(t,x); ylabel('amplitude (a.u.)'); xlabel('time (s)')
title('Original recorded signal'); grid on
% Observaciones:
%   Seal con un proceso que acaba en el segundo 26.05 o 26.06
%   A medida que nos acercamos al tiempo lmite, el proceso se repite cada
%   vez ms rpido

%% Analisis espectral
[Px,f] = pwelch(x, hamming(4096), 0.5, 8192, fs);
figure(2);
plot(f/1000,10*log10(Px)); ylabel('Power Spectral Density (dB)'); xlabel('frequency (kHz)');
xlim([0 25]);
title('PSD of original recorded signal'); grid on

% Observaciones:
%   Seal filtrada paso-baja 20-22 kHz (filtrada probablemente por el propio micrfono) 
%   Picos en torno a 2, 5, 8.3, 10.9, 17.4, 20.08 kHz. Hay otros picos,
%   menos estrechos

%% Espectrograma
% If spectrogram is unavailable in your Octave version, replace it with specgram.
[X,f,ts] = specgram(x, 4096, fs, hamming(2048), 1024);
figure(3);
imagesc(ts,f/1000,10*log10(abs(X)));
axis xy
colormap('jet')
ylim([0 20]);
xlabel('time (s)'); ylabel('frequency (kHz)');
title('Spectrogram of original recorded signal'); grid on


%% Sugerencia: anlisis de seal paso-banda entre 2500 y 4500 (evitando
% armonicos)
B = fir1(400,[2500 4500]/(fs/2));  % filtro paso-banda evito armonico
B = fir1(400,[4600 5400]/(fs/2));  % filtro paso-banda segundo resonancia
B = fir1(400,[1000 3000]/(fs/2));  % filtro paso-banda segundo resonancia
%B = fir1(400,[17200 17600]/(fs/2));  % filtro paso-banda segundo resonancia
%B = fir1(400,[0500 18500]/(fs/2));  % filtro paso-banda
Bo = imag(hilbert(B)); % filtro en cuadratura
y=filter(B,1,x);
yo=filter(Bo,1,x);
a0 = abs(y+1j*yo);  % ENVOLVENTE DE LA SEAL EN LA BANDA SELECCIONADA
figure(4);
plot(t,a0); ylabel('envelope (a.u.)'); xlabel('time (s)')
title('Envelope of the filtered signal'); grid on

%%
a1=resample(a0,1,100);  % submuestreamos en un factor 100
% filtramos paso-alta para quitar la componente de continua de la envolvente 
[Bhp,Ahp]=butter(2,3/(fs/100/2),'high');
a2=filter(Bhp,Ahp,a1);
[A0,f,ts] = specgram(a2, 4096, fs/100, hamming(512), 500);

figure(5)
N2=length(a2);
t2=(0:(N2-1))'/(fs/100);
plot(t2,a2)
ylabel('envelope')
xlabel('time (s)')
title('Envelope of the filtered signal (high-pass filtered)'); grid on

figure(6);
imagesc(ts,f,10*log10(abs(A0)+0.01));
axis xy
colormap('jet')
ylim([0 200]);
xlabel('time (s)'); ylabel('frequency (Hz)');
title('Spectrogram of the envelope (high-pass filtered)'); grid on

figure(7)
imagesc(ts,f,10*log10(abs(A0)+0.01));
axis xy
colormap('jet')
ylim([0 50]);
xlim([18 28]);
xlabel('time (s)'); ylabel('frequency (Hz)');
title('Spectrogram of the envelope (high-pass filtered) - Detail'); grid on

%% Estimacion de la frecuencia del tintineo (mximo del espectrograma)
PSD_A0=(abs(A0)).^2;
[Pm,idxm]=max(PSD_A0);
fmax=f(idxm);
fmax_smooth=fmax;
Df=f(2)-f(1);    % spectral resolution
for k=1:length(ts)
    if idxm(k)>1
        ya=PSD_A0(idxm(k)-1,k)-Pm(k);
        yb=PSD_A0(idxm(k)+1,k)-Pm(k);
        Dx=(ya-yb)/(ya+yb)*0.5;
        fmax_smooth(k)=fmax(k)+Dx*Df;
    end
end
figure(8)
plot(ts,fmax,ts,fmax_smooth)
ylim([0 50]); grid on
xlabel('time (s)')
ylabel('frequency (Hz)')
title('Spectral maximum of envelope')
legend('spectral maximum of FFT','smoothed maximum')


%% for hearing the signals / for printing the plots

if 0
% op=audioplayer(y,fs);       % x: seal original; y: seal en la banda seleccionada
% play(op);

a3=resample(a2,10,1);
a3=a3/max(abs(a3))*0.8;
sound(a3,fs/10)  % Octave playback

figure(1); xlim([21 26.5]); print -dpng -r300 fig1.png
figure(2); print -dpng -r300 fig2.png
figure(3); xlim([21 26.5]); print -dpng -r300 fig3.png
figure(4); xlim([21 26.5]); print -dpng -r300 fig4.png
figure(5); xlim([21 26.5]); print -dpng -r300 fig5.png
figure(6); print -dpng -r300 fig6.png
figure(7); print -dpng -r300 fig7.png
figure(8); print -dpng -r300 fig8.png

end


%% ajuste parametrico de la frecuencia
% fmax_smooth
% f= k (1/(t-t0))^alpha
% alpha = 1/3

lf=log10(fmax_smooth);
t0=26.08;     % valor inicial de t0
x=(log10(t0-ts))';
cond=ts<t0;
[ord,pend,r]=f_regresion(x(cond),lf(cond));
lf_mod=ord+pend*x(cond);

alpha=-pend;
k=10^(ord);
figure(9);
plot(x(cond),lf(cond),'.',x(cond),lf_mod); xlabel('log10(t-t_0)'); ylabel('log10(fmax)'); grid on
fprintf('Estimacin inicial:  alpha = %f    k = %f    t0 = %f    r=%f\n',alpha,k,t0,r)

% segundo analisis, limitando el rango de x
cond=x>-0.5 & x<1 & ts'<t0;
lf=lf(cond);
x=x(cond);
ts1=(ts(cond))';
[ord,pend,r]=f_regresion(x,lf);
lf_mod=ord+pend*x;
alpha=-pend;
k=10^(ord);
f_mod=k.*(1./(t0-ts1)).^alpha;

figure(10);
plot(x,lf,'.',x,lf_mod); xlabel('log10(t-t_0)'); ylabel('log10(fmax)'); grid on
fprintf('Estimacin inicial:  alpha = %f    k = %f    t0 = %f    r=%f\n',alpha,k,t0,r)

figure(11);
plot(ts1,f_mod,'.-')


% buscamos el valor ptimo de t0
t0_opt=-10000;
Error_optimo=1e50;
for t0_test=25:0.05:27
    cond=ts1<t0_test;
    x_test=(log10(t0_test-ts1(cond)));
    lf_test=lf(cond);
    [ord,pend,r]=f_regresion(x_test,lf_test);
    lf_mod=ord+pend*x_test;
    Error=sum((lf_mod-lf_test).^2); 
    if Error<Error_optimo
        Error_optimo=Error;
        t0_opt=t0_test;
        k_opt=10^ord;
        alpha_opt=-pend;
        fprintf('Hemos mejorado!!!\n')
        figure(12);
        plot(x_test,lf_test,'.',x_test,lf_mod); xlabel('log10(t-t_0)'); ylabel('log10(fmax)'); grid on
    end
    fprintf('t0=%f    alpha=%f   Error=%f   r=%f\n',t0_test,-pend,Error,r)  
end

cond=ts<t0_opt;
ts_mod=ts(cond)';
f_mod=k_opt.*(1./(t0_opt-ts_mod)).^alpha_opt;
N=length(ts)-length(ts_mod);
f_mod=[f_mod; ones(N,1)*f_mod(end)];

figure(11)
plot(ts,fmax_smooth,ts,f_mod)

%% demodulacion de fase alrededor de la frecuencia instantanea
f0_mod=interp1(ts,f_mod,t2,'linear','extrap');
w0_mod=2*pi*f0_mod;
phi_mod=cumsum(w0_mod)/fs*100;
exp_comp=exp(-1j*phi_mod);
a2_cbb=a2.*exp_comp;
[Blp,Alp]=butter(2,1.5/(fs/100/2));
a2_cbbf=filtfilt(Blp,Alp,a2_cbb);
phi_i=unwrap(angle(a2_cbbf));

figure(15)
plot(t2,phi_i/pi*180)
Dw=[diff(phi_i)*fs/100; 0];  % error de frecuencia (con respecto a w0) en rad/seg
w_estim=w0_mod+Dw;
f_estim=w_estim/(2*pi);
figure(15)
plot(ts,fmax_smooth,ts,f_mod,t2,f_estim)





