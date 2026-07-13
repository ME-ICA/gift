% demo to show the usage of complex_ICA_EBM
clear;
close all;
clc;

N = 5;             % generate 5 QAM sources
T = 1000;    % the sample size
MAOrder = 5;
K = 4;
SNR=20;
LiteWF=1;  %1 for light version of updating whitening filter
FullWF=0;  %1 for light version of updating whitening filter


clear s;
% QAM sources
for n = 1 : N
    M = 2^n;        % M is the order of QAM, here from 2 to 1024
    x = [0:M-1];    % x the real part symbols
    y = qammod(x,M);    % y is the final symbols, real and imaginary parts
    s(n,:) = randsrc(1,T,[y; ones(1,M)/M]); % generate QAM sources, each symbols have the same probability
    if M == 2
        noise = awgn(s(n,:),SNR,'measured') - s(n,:);    % for real BPSK, added noise is real
        noise = noise.*exp(2*pi*sqrt(-1)*rand(size(noise)));     % random rotate to get complex noise
        s(n,:) = s(n,:) + noise;                        % add complex noise;
    else
        s(n,:) = awgn(s(n,:),SNR,'measured');    % by default, add complex Gaussian noise;
    end
    
    %             scatterplot(s(n,:))     % show the scatter plot;
end

% color the data
for n = 1 : N
    s(n,:) = filter(randn(1,MAOrder)+sqrt(-1)*randn(1,MAOrder), 1, s(n,:));
    s(n,:) = s(n,:) - mean(s(n,:));
    s(n,:) = sqrt(T)*s(n,:)/norm(s(n,:));
end

% mixing
A = randn(N,N)+sqrt(-1)*randn(N,N);
x = A*s;

% separation
[W_CERBML, ~, ~] = CERBM(x, LiteWF);

[W_CERBM, ~, ~] = CERBM(x, FullWF);

[W_CEBM, ~, ~] = CEBM(x);

% showing results
G = abs(W_CERBM*A);
for n=1:N
    G(n,:)=G(n,:)/(max(abs(G(n,:))));
end
G=kron(G,ones(50,50));
figure;
imshow(abs(G))
title('Confusion matrix for CERBM','FontSize',15,'FontName','Arial');

G = abs(W_CERBML*A);
for n=1:N
    G(n,:)=G(n,:)/(max(abs(G(n,:))));
end
G=kron(G,ones(50,50));
figure;
imshow(abs(G))
title('Confusion matrix for CERBM-L','FontSize',15,'FontName','Arial');

G = abs(W_CEBM*A);
for n=1:N
    G(n,:)=G(n,:)/(max(abs(G(n,:))));
end
G=kron(G,ones(50,50));
figure;
imshow(abs(G))
title('Confusion matrix for CEBM','FontSize',15,'FontName','Arial');

