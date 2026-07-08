function [W, Ahat, Shat] = CERBM( X, Lite, p )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CERBM: complex entropy rate bound minimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
% X:           mixtures X
% Lite:        1 for lite version CERBM, 0 for full version CERBM, optional.
% p:           whitening filter order, optional.
% Outputs:     
% W:           demixing matrix W
% Ahat:        estimation of mixing matrix
% Shat:        estimation of sources
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Reference
% Geng-Shen Fu, Ronald Phlypo, Matthew Anderson, and Tulay Adali, 
% "Complex Independent Component Analysis using Three Types of Diversity: 
% Non-Gaussianity, Nonwhiteness, and Noncircularity," in IEEE Trans. on
% Signal Processing, vol. 63, no. 3, pp. 794-805, Feb. 2015. 
%
% Geng-Shen Fu, Ronald Phlypo, Matthew Anderson, Xi-Lin Li, and Tulay Adali, 
% "An efficient entropy rate estimator for complex-valued signal processing:
% Application to ICA," in Acoustics, Speech and Signal Processing (ICASSP),
% 2014 IEEE International Conference on, May 2014, pp. 6216?6220. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program by Geng-Shen Fu
%
%
% Copyright (C) 2023 MLSP Lab
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% Please make sure you include the references in any related work.
% 
% A copy of the GNU General Public License can be found at 
% <https://mlsp.umbc.edu/resources.html>.
% If not, see <https://www.gnu.org/licenses/>.
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
show_cost = 0;
InitCEBM = 1;

[N,T] = size(X);
% read the input arguments and initialization
if nargin < 3
    p = round(log10(T)*1.4);
end
if nargin < 2
    Lite = 0;
end
    
% Whitening
[Xc, P] = pre_processing( X );

initCost = inf;
if InitCEBM == 1
    [W initCost Ahat Shat] = CEBM(Xc);     % use ICA-EBM to provide the initial guess
else
%     W = eye(N, N);
    W = randn(N)+sqrt(-1)*randn(N);
    W = inv(sqrtm(W*W'))*W;
end
if p == 1               % only exploiting the non-Gaussianity and noncircular for separation
    W = W*P;
    return;
end

% load 8 nonlinearities, but we only use 4 of them
K_real = 8;
load nf_table
global real_nf1 real_nf2 real_nf3 real_nf4 real_nf5 real_nf6 real_nf7 real_nf8;
real_nf1 = nf1;     % x^4
real_nf2 = nf2;     % |x|
real_nf3 = nf3;     % |x|/(1+|x|)
real_nf4 = nf4;     % |x|/(1+x^2)
real_nf5 = nf5;     % x|x|/(10+|x|)
real_nf6 = nf6;     % x/(1+|x|)
real_nf7 = nf7;     % x/(1+x^2)
real_nf8 = nf8;


Xc0 = [zeros(N, p-1), Xc];
idxXtilde = kron(ones(1,T),[p:-1:1]) + kron([0:T-1],ones(1,p));
Xtilde = Xc0(:,idxXtilde);
idxXtilde = [];
for i=1:p
    idxXtilde = [idxXtilde [p-i+1:T+p-i]];
end
XtildeColtmp = Xc0(:,idxXtilde);
XtildeCol=reshape(XtildeColtmp,N*T,p);
clear idxXtilde;
clear XtildeColtmp;
clear Xc0;

tolerance = 1e-7;   % stopping condition
iterUpdateWF = 3;
last_W = W;
best_W = W;
best_a = zeros(2*p,2*p);
best_a(1,:) = 1;
best_Grad = zeros(N,N);
C = Xc*Xc.'/T;      % C is the pseudo covariance matrix
maxiter = 100;
Cost = zeros(maxiter,1);
min_cost = initCost;
cost_increase_counter = 0;
cost_decrease_counter = 0;
mu=1/50;
max_cost_increase_number = 2;
max_cost_decrease_number = 3;
Jc = sqrt(-1);
for iter = 1 : maxiter
    Yc = W*Xc;
    H = decoupling_trick (W);
    
    Cost(iter) = Cost(iter) - 2*log(abs(det(W)));
    for n = 1 : N
        Wn = W(n,:)';
        if iter<=5
            %Estimate whitening filter without intial guess
            [a(:,n), zC, costWF] = lfcc(Yc(n,:), p, [], Lite);
        elseif (mod(iter,iterUpdateWF)==1)
            %Estimate whitening filter using previous estimates as intial guess
            [a(:,n), zC, costWF] = lfcc(Yc(n,:), p, a(:,n), Lite);
        else
            % remove DC
            tmpYc=Yc(n,:)-mean(Yc(n,:));
            tmpYcM0 = convmtx(tmpYc,p);
            tmpYcM = tmpYcM0(:, 1:T);
            augYcM=[tmpYcM;conj(tmpYcM)];
            zC = (a(:,n)'*augYcM).';
        end
        
        %Calculte cost and gradient
        [Costi Gradi] = calCostGrad(a(:,n), zC, Xtilde);
        
        grad = Gradi - H(:,n)/(Wn'*H(:,n));
        grad = grad - Wn'*grad/(Wn'*Wn)*Wn;
        
        gradM(:,n) = grad / norm(grad);
        
        %Calculate new W
        Wn = Wn - mu*gradM(:,n);
        W(n,:) = Wn' / norm(Wn);
        Cost(iter) = Cost(iter) + Costi;
    end
    
    if Cost(iter) < min_cost
        min_cost = Cost(iter);
        best_W = last_W;
        best_Grad = gradM;
        best_a = a;
        cost_increase_counter = 0;
        cost_decrease_counter = cost_decrease_counter + 1;
        if cost_decrease_counter > max_cost_decrease_number
            mu = mu * 1.5;
            cost_decrease_counter=0;
        end
    else
        cost_increase_counter = cost_increase_counter + 1;
        cost_decrease_counter=0;
    end
    
    if cost_increase_counter > max_cost_increase_number
        if sum(abs(abs(diag(W*last_W'))-1)) < N*tolerance
            break;
        else
            mu = mu/2;
            cost_increase_counter = 0;
            W = best_W;
            gradM = best_Grad;
            a = best_a;
            %search from best W using smaller mu and best grad
            W=W-(gradM*mu)';
            W=diag(1./sqrt(diag(W*W')))*W;
            
            last_W = W;
            continue;
        end
    end
    last_W = W;
end

W = best_W;
if show_cost
    MyFontSize = 12;
    MyMarkerSize = 10;
    MyLineWidth = 1.3;
    
    file_name='/Users/fugs/Dropbox/fgs/ICA/CERBM/Latest/result/Latest/CostCERBM';
    figure;
    plot(Cost(1:iter), '-b', 'MarkerSize', MyMarkerSize, 'linewidth', MyLineWidth);
    xlabel('Number of iterations','FontSize',MyFontSize,'FontName','Arial')
    ylabel('Cost','FontSize',MyFontSize,'FontName','Arial')
    title('CERBM','FontSize',MyFontSize,'FontName','Arial');
    set(gca,'FontSize',MyFontSize);
    set(gca,'FontName','Arial');
    
    print('-depsc', file_name);
    eps2pdf([file_name, '.eps'], '/usr/local/bin/gs');
    hgsave(file_name);
end

W = W*P;

% if you want to get the sources, it is
Ahat = inv(W);
Shat = W*X;



function [Cost Newton] = calCostNewton(a, zC, XtildeCol, Hn, Wn)
global real_nf1 real_nf2 real_nf3 real_nf4 real_nf5 real_nf6 real_nf7 real_nf8;

Jc = sqrt(-1);
Cost=0;
p = size(a, 1)/2;
N = size(Wn, 1);
T = size(zC, 1);

zR = real(zC);
zI = imag(zC);
sigmaR2=var(zR);
sigmaI2=var(zI);
u=zR/sqrt(sigmaR2);
v=zI/sqrt(sigmaI2);
Cost = log(2*pi)+1+0.5*log(sigmaR2*sigmaI2);

a1 = a(1:p);
a2 = a(p+1:end);

% evaluate the upper bound of negentropy of the nth component
% we only need to calculate these quantities once
uu = u.*u;
sign_u = sign(u);
abs_u = sign_u.*u;
inv_pabs_u = 1./(1+abs_u);
inv_pabs_uu = 1./(1+uu);
inv_p10abs_u = 1./(10+abs_u);

vv = v.*v;
sign_v = sign(v);
abs_v = sign_v.*v;
inv_pabs_v = 1./(1+abs_v);
inv_pabs_vv = 1./(1+vv);
inv_p10abs_v = 1./(10+abs_v);

[NE_Boundu EGu] = estEntBound(u, uu, abs_u, inv_pabs_u, inv_pabs_uu, inv_p10abs_u, T);
[NE_Boundv EGv] = estEntBound(v, vv, abs_v, inv_pabs_v, inv_pabs_vv, inv_p10abs_v, T);

% select the tightest bound
[max_NEu, max_iu] = max( NE_Boundu );
[max_NEv, max_iv] = max( NE_Boundv );
Cost = real(Cost - max_NEu - max_NEv);

XtildeCol = conj(XtildeCol);
XtildeAplus = reshape(XtildeCol*(a1+conj(a2)), N, T);
XtildeAminus = reshape(XtildeCol*(a1-conj(a2)), N, T);

% calculation of the gradient: dsigmaR2/dw and dsigmaI2/dw
gradSigmaR2 = XtildeAplus*zR/T;
gradSigmaI2 = XtildeAminus*zI*Jc/T;

% calculation of the hessian: d^2sigmaR2/dww^T and d^2sigmaI2/dw
hessSigmaR2T = XtildeAplus * (XtildeAplus.')/T/2;
hessSigmaR2H = XtildeAplus * (XtildeAplus')/T/2;
hessSigmaI2T = XtildeAminus * (XtildeAminus.')/T/(-2);
hessSigmaI2H = XtildeAminus * (XtildeAminus')/T/2;

grad = gradSigmaR2/sigmaR2/2 + gradSigmaI2/sigmaI2/2;
H1 = hessSigmaR2T/sigmaR2/2 - gradSigmaR2*(gradSigmaR2.')/(sigmaR2^2)/2 + hessSigmaI2T/sigmaI2/2 - gradSigmaI2*(gradSigmaI2.')/(sigmaI2^2)/2;
H2 = hessSigmaR2H/sigmaR2/2 - gradSigmaR2*(gradSigmaR2')/(sigmaR2^2)/2 + hessSigmaI2H/sigmaI2/2 - gradSigmaI2*(gradSigmaI2')/(sigmaI2^2)/2;

%gu and gv are first order derivative
%ggu and ggv are second order derivative
switch max_iu
    case 1
        EGu(1) = max( min( EGu(1), real_nf1.max_EGx ), real_nf1.min_EGx );
        gu=(4*u.*uu);
        ggu=(12*uu);
        vGu=simplified_ppval( real_nf1.pp_slope, EGu(1) );
    case 3
        EGu(3) = max( min( EGu(3), real_nf3.max_EGx ), real_nf3.min_EGx );
        gu=( sign_u.*inv_pabs_u.^2 );
        ggu=( (-2)*(inv_pabs_u.^3) );
        vGu=simplified_ppval( real_nf3.pp_slope, EGu(3) );
    case 5
        EGu(5) = max( min( EGu(5), real_nf5.max_EGx ), real_nf5.min_EGx );
        gu=( abs_u.*(20+abs_u).*inv_p10abs_u.^2 );
        ggu=( 200*sign_u.*(inv_p10abs_u.^3) );
        vGu=simplified_ppval( real_nf5.pp_slope, EGu(5) );
    case 7
        EGu(7) = max( min( EGu(7), real_nf7.max_EGx ), real_nf7.min_EGx );
        gu=( (1-uu).*inv_pabs_uu.^2 );
        ggu=( 2*u.*(uu-3).*(inv_pabs_uu.^3) );
        vGu=simplified_ppval( real_nf7.pp_slope, EGu(7) );
    otherwise
        ;
end
grad = grad - vGu*(XtildeAplus*gu)/sqrt(sigmaR2)/2/T;
H1 = H1 - vGu*((repmat(ggu',N,1).*XtildeAplus)*(XtildeAplus.'))/sigmaR2/4/T;
H2 = H2 - vGu*((repmat(ggu',N,1).*XtildeAplus)*(XtildeAplus'))/sigmaR2/4/T;
% grad = grad - vGu*(XtildeAplus*gu)/2/T;
% H1 = H1 - vGu*((repmat(ggu',N,1).*XtildeAplus)*(XtildeAplus.'))/4/T;
% H2 = H2 - vGu*((repmat(ggu',N,1).*XtildeAplus)*(XtildeAplus'))/4/T;

switch max_iv
    case 1
        EGv(1) = max( min( EGv(1), real_nf1.max_EGx ), real_nf1.min_EGx );
        gv=(4*v.*vv);
        ggv=(12*vv);
        vGv=simplified_ppval( real_nf1.pp_slope, EGv(1) );
    case 3
        EGv(3) = max( min( EGv(3), real_nf3.max_EGx ), real_nf3.min_EGx );
        gv=( sign_v.*inv_pabs_v.^2 );
        ggv=( (-2)*(inv_pabs_v.^3) );
        vGv=simplified_ppval( real_nf3.pp_slope, EGv(3) );
    case 5
        EGv(5) = max( min( EGv(5), real_nf5.max_EGx ), real_nf5.min_EGx );
        gv=( abs_v.*(20+abs_v).*inv_p10abs_v.^2 );
        ggv=( 200*sign_v.*(inv_p10abs_v.^3) );
        vGv=simplified_ppval( real_nf5.pp_slope, EGv(5) );
    case 7
        EGv(7) = max( min( EGv(7), real_nf7.max_EGx ), real_nf7.min_EGx );
        gv=( (1-vv).*inv_pabs_vv.^2 );
        ggv=( 2*v.*(vv-3).*(inv_pabs_vv.^3) );
        vGv=simplified_ppval( real_nf7.pp_slope, EGv(7) );
    otherwise
        ;
end
grad = grad - Jc*vGv*(XtildeAminus*gv)/sqrt(sigmaI2)/2/T;
H1 = H1 + vGv*((repmat(ggv',N,1).*XtildeAminus)*(XtildeAminus.'))/sigmaI2/4/T;
H2 = H2 - vGv*((repmat(ggv',N,1).*XtildeAminus)*(XtildeAminus'))/sigmaI2/4/T;
% grad = grad - Jc*vGv*(XtildeAminus*gv)/2/T;
% H1 = H1 + vGv*((repmat(ggv',N,1).*XtildeAminus)*(XtildeAminus.'))/4/T;
% H2 = H2 - vGv*((repmat(ggv',N,1).*XtildeAminus)*(XtildeAminus'))/4/T;

grad = grad - conj(Hn)/(Hn'*Wn);
H1 = H1 + conj(Hn*Hn.')/(Hn'*Wn)^2;

Newton = inv(conj(H2)-conj(H1)*inv(H2)*H1)*(conj(grad)-conj(H1)*inv(H2)*grad);
i=1;


function [Cost grad] = calCostGrad(a, zC, Xtilde)
global real_nf1 real_nf2 real_nf3 real_nf4 real_nf5 real_nf6 real_nf7 real_nf8;

Jc = sqrt(-1);
Cost=0;
p = size(a, 1)/2;
T = size(zC, 1);

sigmaR2=var(real(zC));
sigmaI2=var(imag(zC));
u=real(zC)/sqrt(sigmaR2);
v=imag(zC)/sqrt(sigmaI2);
Cost = log(2*pi)+1+0.5*log(sigmaR2*sigmaI2);

a1 = a(1:p);
a2 = a(p+1:end);

% evaluate the upper bound of negentropy of the nth component
% we only need to calculate these quantities once
uu = u.*u;
sign_u = sign(u);
abs_u = sign_u.*u;
inv_pabs_u = 1./(1+abs_u);
inv_pabs_uu = 1./(1+uu);
inv_p10abs_u = 1./(10+abs_u);

vv = v.*v;
sign_v = sign(v);
abs_v = sign_v.*v;
inv_pabs_v = 1./(1+abs_v);
inv_pabs_vv = 1./(1+vv);
inv_p10abs_v = 1./(10+abs_v);

[NE_Boundu EGu] = estEntBound(u, uu, abs_u, inv_pabs_u, inv_pabs_uu, inv_p10abs_u, T);
[NE_Boundv EGv] = estEntBound(v, vv, abs_v, inv_pabs_v, inv_pabs_vv, inv_p10abs_v, T);

% select the tightest bound
[max_NEu, max_iu] = max( NE_Boundu );
[max_NEv, max_iv] = max( NE_Boundv );
Cost = real(Cost - max_NEu - max_NEv);

% calculation of the gradient
gradSigmaR2 = Xtilde*kron(zC, conj(a1)+a2)/T/2 + Xtilde*kron(conj(zC), conj(a1)+a2)/T/2;
gradSigmaI2 = Xtilde*kron(conj(zC), conj(a1)-a2)/T/2 - Xtilde*kron(zC, conj(a1)-a2)/T/2;

grad = gradSigmaR2/sigmaR2/2 + gradSigmaI2/sigmaI2/2;
aaplus = conj(a(1:p))+a(p+1:end);
aaminus = conj(a(1:p))-a(p+1:end);

switch max_iu
    case 1
        EGu(1) = max( min( EGu(1), real_nf1.max_EGx ), real_nf1.min_EGx );
        gu=(4*u.*uu);
        vGu=simplified_ppval( real_nf1.pp_slope, EGu(1) );
    case 3
        EGu(3) = max( min( EGu(3), real_nf3.max_EGx ), real_nf3.min_EGx );
        gu=( sign_u.*inv_pabs_u.^2 );
        vGu=simplified_ppval( real_nf3.pp_slope, EGu(3) );
    case 5
        EGu(5) = max( min( EGu(5), real_nf5.max_EGx ), real_nf5.min_EGx );
        gu=( abs_u.*(20+abs_u).*inv_p10abs_u.^2 );
        vGu=simplified_ppval( real_nf5.pp_slope, EGu(5) );
    case 7
        EGu(7) = max( min( EGu(7), real_nf7.max_EGx ), real_nf7.min_EGx );
        gu=( (1-uu).*inv_pabs_uu.^2 );
        vGu=simplified_ppval( real_nf7.pp_slope, EGu(7) );
    otherwise
        ;
end
%E{g(u)*(dz_R/dw*)/sigmaR}
grad = grad - vGu*(Xtilde*kron(gu, conj(a1)+a2)/T)/2/sqrt(sigmaR2);
%E{g(u)*(dz_R/dw*)/sigmaR + g(u)*(dsigmaR2/dw*)*z_R/2/sigmaR3}
% grad = grad - vGu*(Xtilde*kron(gu, conj(a1)+a2)/T)/2/sqrt(sigmaR2) - vGu*(gu'*real(zC))*gradSigmaR2/2/sigmaR2/sqrt(sigmaR2)/T;

switch max_iv
    case 1
        EGv(1) = max( min( EGv(1), real_nf1.max_EGx ), real_nf1.min_EGx );
        gv=(4*v.*vv);
        vGv=simplified_ppval( real_nf1.pp_slope, EGv(1) );
    case 3
        EGv(3) = max( min( EGv(3), real_nf3.max_EGx ), real_nf3.min_EGx );
        gv=( sign_v.*inv_pabs_v.^2 );
        vGv=simplified_ppval( real_nf3.pp_slope, EGv(3) );
    case 5
        EGv(5) = max( min( EGv(5), real_nf5.max_EGx ), real_nf5.min_EGx );
        gv=( abs_v.*(20+abs_v).*inv_p10abs_v.^2 );
        vGv=simplified_ppval( real_nf5.pp_slope, EGv(5) );
    case 7
        EGv(7) = max( min( EGv(7), real_nf7.max_EGx ), real_nf7.min_EGx );
        gv=( (1-vv).*inv_pabs_vv.^2 );
        vGv=simplified_ppval( real_nf7.pp_slope, EGv(7) );
    otherwise
        ;
end
%E{g(v)*(dz_I/dw*)/sigmaI}
grad = grad + vGv*Jc*(Xtilde*kron(gv, conj(a1)-a2))/T/2/sqrt(sigmaI2);
%E{g(v)*(dz_I/dw*)/sigmaI + g(v)*(dsigmaI2/dw*)*z_I/2/sigmaI3}
% grad = grad + vGv*Jc*(Xtilde*kron(gv, conj(a1)-a2))/T/2/sqrt(sigmaI2) - vGv*(gv'*imag(zC))*gradSigmaI2/2/sigmaI2/sqrt(sigmaI2)/T;




function [a, y, min_cost] = lfcc(x, p, a0, Lite)
% Return the linear filtering coefficient (LFC) with length p for entropy
% rate estimation, and the estimated entropy rate
% Inputs
% p is the filter length
% a0 is the intial guess
% Outputs
% a is the filter coefficients
% min_cost is the entropy rate estimation
global real_nf1 real_nf2 real_nf3 real_nf4 real_nf5 real_nf6 real_nf7 real_nf8;

tolerance = 1e-4;
T = length(x);
Da=diag([1, zeros(1, p-1), -1, zeros(1, p-1)]);

% remove DC
x=x-mean(x);

X0 = convmtx(x,p);      
X = X0(:, 1:T);         % if use the whole convmtx, outliers arise, so remove the tail
% % remove DC
% Xmean=mean(X,2);
% X = X - Xmean*ones(1,T);
Rc = X*X'/T;
Rp = X*X.'/T;
RaugH=[Rc Rp; conj(Rp) conj(Rc)];
RaugT=[Rp Rc; conj(Rc) conj(Rp)];
X=[X;conj(X)];

if Lite == 1
    % use Gaussain cost to provide the initial guess
    [V,D] = eig(inv(RaugH)*Da);
    [maxD, idxD]=max(diag(real(D)));
    a=V(:,idxD);
    y = (a'*X).';
    min_cost = inf;
    return;
end

% initialize a
if isempty(a0)
    % use Gaussain cost to provide the initial guess
    [V,D] = eig(inv(RaugH)*Da);
    [maxD, idxD]=max(diag(real(D)));
    a=V(:,idxD);
else
    a = a0;
end

% Load 8 measuring functions. 
K = 8;

min_cost = inf;
best_a = a;
last_a = a;
min_mu = 1/1024;
if isempty(a0)
    max_iter = 100;
    mu = 8*min_mu;
else
    max_iter = 100;
    mu = 16*min_mu;
end
max_cost_decrease=2;
cost_increase_counter=0;
cost_decrease_counter=0;
Cost = zeros(max_iter,1);
for iter = 1 : max_iter
    
    [b, G] = cnstd_and_gain_c (a, Da);
    a = a/G;
    
    y = a'*X;
    sigmaR2 = var(real(y));
    sigmaI2 = var(imag(y));
%     sigmaR2 = (a'*RaugT*conj(a)+a'*RaugH*a+conj(a'*RaugH*a)+conj(a'*RaugT*conj(a)))/4;
%     sigmaI2 = -(a'*RaugT*conj(a)-a'*RaugH*a-conj(a'*RaugH*a)+conj(a'*RaugT*conj(a)))/4;
    zR=real(y)/sqrt(sigmaR2);
    zI=imag(y)/sqrt(sigmaI2);
%     y = y/sqrt(sigma2);     % y is always the normalized y here
    Cost(iter) = 0.5*log(((2*pi)^2)*sigmaR2*sigmaI2)+1;
    
	% evaluate the upper bound of negentropy of the nth component
        
    % we only need to calculate these quantities once
    zzR = zR.*zR;
    sign_zR = sign(zR);
    abs_zR = sign_zR.*zR;
    inv_pabs_zR = 1./(1+abs_zR);
    inv_pabs_zzR = 1./(1+zzR);
    inv_p10abs_zR = 1./(10+abs_zR);
    
    zzI = zI.*zI;
    sign_zI = sign(zI);
    abs_zI = sign_zI.*zI;
    inv_pabs_zI = 1./(1+abs_zI);
    inv_pabs_zzI = 1./(1+zzI);
    inv_p10abs_zI = 1./(10+abs_zI);
        
    [NE_BoundR EGxR] = estEntBound(zR, zzR, abs_zR, inv_pabs_zR, inv_pabs_zzR, inv_p10abs_zR, T);
    [NE_BoundI EGxI] = estEntBound(zI, zzI, abs_zI, inv_pabs_zI, inv_pabs_zzI, inv_p10abs_zI, T);
        
	% select the tightest bound
    [max_NER, max_iR] = max( NE_BoundR );
    [max_NEI, max_iI] = max( NE_BoundI );
    Cost(iter) = real(Cost(iter) - max_NER - max_NEI);
    
    if Cost(iter) < min_cost
        cost_increase_counter = 0;
        min_cost = Cost(iter);
        best_a = a;
        cost_decrease_counter=cost_decrease_counter+1;
        if cost_decrease_counter>max_cost_decrease
            cost_decrease_counter=0;
            mu=mu*1.5;
        end;
    else
        cost_increase_counter = cost_increase_counter + 1;
        cost_decrease_counter=0;
    end
    
    if cost_increase_counter > 0
        if mu > min_mu
            mu = mu/2;
            cost_increase_counter = 0;
            a = best_a;
            a = a - mu*grad;
            last_a = best_a;
            continue;
        else
            break;
        end
    end
    
    last_a = a;
    
    grad = (RaugT*conj(a)+RaugH*a)/sigmaR2/4 - (RaugT*conj(a)-RaugH*a)/sigmaI2/4;
    
    switch max_iR
        case 1
        	EGxR(1) = max( min( EGxR(1), real_nf1.max_EGx ), real_nf1.min_EGx );
            gzR=(4*zR.*zzR)';
            vGzR=simplified_ppval( real_nf1.pp_slope, EGxR(1) );            
        case 3
            EGxR(3) = max( min( EGxR(3), real_nf3.max_EGx ), real_nf3.min_EGx );
            gzR=( sign_zR.*inv_pabs_zR.^2 )';
            vGzR=simplified_ppval( real_nf3.pp_slope, EGxR(3) );
        case 5
            EGxR(5) = max( min( EGxR(5), real_nf5.max_EGx ), real_nf5.min_EGx );
            gzR=( abs_zR.*(20+abs_zR).*inv_p10abs_zR.^2 )';
            vGzR=simplified_ppval( real_nf5.pp_slope, EGxR(5) );
        case 7
        	EGxR(7) = max( min( EGxR(7), real_nf7.max_EGx ), real_nf7.min_EGx );
            gzR=( (1-zzR).*inv_pabs_zzR.^2 )';
            vGzR=simplified_ppval( real_nf7.pp_slope, EGxR(7) );
        otherwise
        	;
    end
	grad = grad - vGzR*(X*gzR/T/2/sqrt(sigmaR2) - zR*gzR*(RaugT*conj(a)+RaugH*a)/T/4/sigmaR2);
            
    switch max_iI
        case 1
        	EGxI(1) = max( min( EGxI(1), real_nf1.max_EGx ), real_nf1.min_EGx );
            gzI=(4*zI.*zzI)';
            vGzI=simplified_ppval( real_nf1.pp_slope, EGxI(1) );            
        case 3
            EGxI(3) = max( min( EGxI(3), real_nf3.max_EGx ), real_nf3.min_EGx );
            gzI=( sign_zI.*inv_pabs_zI.^2 )';
            vGzI=simplified_ppval( real_nf3.pp_slope, EGxI(3) );
        case 5
            EGxI(5) = max( min( EGxI(5), real_nf5.max_EGx ), real_nf5.min_EGx );
            gzI=( abs_zI.*(20+abs_zI).*inv_p10abs_zI.^2 )';
            vGzI=simplified_ppval( real_nf5.pp_slope, EGxI(5) );
        case 7
        	EGxI(7) = max( min( EGxI(7), real_nf7.max_EGx ), real_nf7.min_EGx );
            gzI=( (1-zzI).*inv_pabs_zzI.^2 )';
            vGzI=simplified_ppval( real_nf7.pp_slope, EGxI(7) );
        otherwise
        	;
    end
	grad = grad - vGzI*(-sqrt(-1)*X*gzI/T/2/sqrt(sigmaI2) + zI*gzI*(RaugT*conj(a)-RaugH*a)/T/4/sigmaI2);
                    
    grad = grad - grad'*b*b/(b'*b);
    grad = sqrt(sigmaR2+sigmaI2) * grad / norm(grad);
    a = a - mu*grad;
        
end
a = best_a;

y = (a'*X).';

% % %for debug fgs: i.i.d.
% % a = [1; zeros(2*p-1,1)];
% file_name = '/Users/fugs/Dropbox/fgs/ICA/CERBM/Latest/result/Latest/CostIterWhiteningFilter1'
% MyFontSize = 15;
% MyMarkerSize = 12;
% MyLineWidth = 2.4;
% 
% plot(Cost(Cost~=0), '-k', 'MarkerSize', MyMarkerSize, 'linewidth', MyLineWidth);
% ylabel('Cost','FontSize',MyFontSize,'FontName','Arial');
% xlabel('Number of iterations','FontSize',MyFontSize,'FontName','Arial');
% set(gca,'FontSize',MyFontSize);
% set(gca,'FontName','Arial');
% 
% print('-depsc', file_name);
% eps2pdf([file_name, '.eps'], '/usr/local/bin/gs');
% hgsave(file_name);

i=1;



function [NE_Bound EGx] = estEntBound(y, yy, abs_y, inv_pabs_y, inv_pabs_yy, inv_p10abs_y, T)
% load nf_table
global real_nf1 real_nf2 real_nf3 real_nf4 real_nf5 real_nf6 real_nf7 real_nf8;
K=8;
NE_Bound = zeros(K,1);
EGx = zeros(K,1);
% G1(x) = x^4
EGx(1) = sum( yy.*yy )/T;
if EGx(1)<real_nf1.min_EGx
    NE_Bound(1) =  simplified_ppval( real_nf1.pp_slope, real_nf1.min_EGx ) * ( EGx(1) - real_nf1.min_EGx );
    NE_Bound(1) = simplified_ppval( real_nf1.pp, real_nf1.min_EGx ) + abs( NE_Bound(1) );
elseif EGx(1)>real_nf1.max_EGx
    % NE_Bound(1) =  simplified_ppval( real_nf1.pp_slope, real_nf1.max_EGx ) * ( EGx(1) - real_nf1.max_EGx );
    % NE_Bound(1) = simplified_ppval( real_nf1.pp, real_nf1.max_EGx ) + abs( NE_Bound(1) );
    NE_Bound(1) = 0;
else
    NE_Bound(1) =  simplified_ppval( real_nf1.pp, EGx(1) );
end
        
% G3 = @(x) abs(x)./( 1+abs(x) );
EGx(3) = 1-sum( inv_pabs_y )/T;
if EGx(3)<real_nf3.min_EGx
    NE_Bound(3) =  simplified_ppval( real_nf3.pp_slope, real_nf3.min_EGx ) * ( EGx(3) - real_nf3.min_EGx );
    NE_Bound(3) = simplified_ppval( real_nf3.pp, real_nf3.min_EGx ) + abs( NE_Bound(3) );
elseif EGx(3)>real_nf3.max_EGx
    NE_Bound(3) =  simplified_ppval( real_nf3.pp_slope, real_nf3.max_EGx ) * ( EGx(3) - real_nf3.max_EGx );
    NE_Bound(3) = simplified_ppval( real_nf3.pp, real_nf3.max_EGx ) + abs( NE_Bound(3) );
else
    NE_Bound(3) =  simplified_ppval( real_nf3.pp, EGx(3) );
end
        
% G5 = @(x) x*|x|/(10+|x|);
EGx(5) = sum( y.*abs_y.*inv_p10abs_y )/T;
if EGx(5)<real_nf5.min_EGx
    NE_Bound(5) =  simplified_ppval( real_nf5.pp_slope, real_nf5.min_EGx ) * ( EGx(5) - real_nf5.min_EGx );
    NE_Bound(5) = simplified_ppval( real_nf5.pp, real_nf5.min_EGx ) + abs( NE_Bound(5) );
elseif EGx(5)>real_nf5.max_EGx
    NE_Bound(5) =  simplified_ppval( real_nf5.pp_slope, real_nf5.max_EGx ) * ( EGx(5) - real_nf5.max_EGx );
    NE_Bound(5) = simplified_ppval( real_nf5.pp, real_nf5.max_EGx ) + abs( NE_Bound(5) );
else
    NE_Bound(5) =  simplified_ppval( real_nf5.pp, EGx(5) );
end
        
% G7 = @(x) x/(1+x^2);
EGx(7) = sum( y.*inv_pabs_yy )/T;
if EGx(7)<real_nf7.min_EGx
    NE_Bound(7) =  simplified_ppval( real_nf7.pp_slope, real_nf7.min_EGx ) * ( EGx(7) - real_nf7.min_EGx );
    NE_Bound(7) = simplified_ppval( real_nf7.pp, real_nf7.min_EGx ) + abs( NE_Bound(7) );
elseif EGx(7)>real_nf7.max_EGx
    NE_Bound(7) =  simplified_ppval( real_nf7.pp_slope, real_nf7.max_EGx ) * ( EGx(7) - real_nf7.max_EGx );
    NE_Bound(7) = simplified_ppval( real_nf7.pp, real_nf7.max_EGx ) + abs( NE_Bound(7) );
else
    NE_Bound(7) =  simplified_ppval( real_nf7.pp, EGx(7) );
end


function [b, G] = cnstd_and_gain_c (a, Da)
% return constraint direction used for calculating projected gradient and Gain of filter a

b=2*Da*a;
G=sqrt(a'*Da*a);


function [H] = decoupling_trick (W)
N = size(W, 1);

H = zeros(N,N);
for n = 1:N
    [Q,~]=qr(W([(1:n-1) (n+1:N)],:)');
    H(:,n)=Q(:,end); % h should be orthogonal to W(nout,:,k)'
end



function [X, P] = pre_processing( X )
% pre-processing program
[N,T] = size(X);
% remove DC
Xmean=mean(X,2);
X = X - Xmean*ones(1,T);    

% spatio pre-whitening 1 
R = X*X'/T;                 
P = inv_sqrtmH(R);  %P = inv(sqrtm(R));
X = P*X;



function A = inv_sqrtmH( B )
% 
[V,D] = eig(B);
d = diag(D);
d = 1./sqrt(d);
A = V*diag(d)*V';


function v = simplified_ppval(pp,xs)
% a simplified version of ppval

b = pp.breaks;
c = pp.coefs;
l = pp.pieces;
k = 4;  % k = pp.order;
dd = 1; % dd = pp.dim;

% find index
index=0;
middle_index=0;
if xs>b(l)
    index = l;
else if xs<b(2)
        index = 1;
    else 
        
        low_index = 1;
        high_index = l;
        while 1
            
            middle_index = round( (low_index + high_index)/2 );
            if b( middle_index ) > xs
                high_index = middle_index;
            else
                low_index = middle_index;
            end
            
            if low_index == high_index-1
                index = low_index;
                break;
            end
            
        end
        
    end
end
    

% now go to local coordinates ...
xs = xs-b(index);

% ... and apply nested multiplication:
   v = c(index,1);
   for i=2:k
      v = xs.*v + c(index,i);
   end
