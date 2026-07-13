%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reference:
%Mike Novey and T. Adali, "COMPLEX FIXED-POINT ICA ALGORITHM FOR 
%            SEPARATION OF QAM SOURCES USING GAUSSIAN MIXTURE MODEL" in
% IEEE Conf. ICASSP 2007,
% June 21, 2009
% ICA algorithm for QAM signals
% where numStarsIn is number points in constellation and
% siggSq is variance of Gaussian kernels
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
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details. 
% Please make sure you include the references in any related work.
% 
% A copy of the GNU General Public License can be found at 
% <https://mlsp.umbc.edu/resources.html>.
% If not, see <https://www.gnu.org/licenses/>.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Ahat, shat] = CQAMsym(xold,  numStarsIn, siggSq)


global numStars;
global U;
numStars = numStarsIn;
 U = zeros(numStars,1);
%%Set up vector of constallation centers (means) of kernels
if numStars < 16
     for n = 0:numStars - 1
            U(n+1,1) = sin((2*pi/numStars) * n) + j*cos((2*pi/numStars) * n);
      end
  else   %%qam 16
      index = 1;
      for r = 1:4
          for c=1:4
            U(index,1) = (-1 + (r-1)*2/3 ) + j*(-1 + (c-1)*2/3);  
            index = index+1;   
          end
      end
end

mixCoef = ones(numStars,1)*1/numStars;
K = 1;
%sigg = .3;
 tol = 1e-5;
 

 [n,m] = size(xold);

[Ex, Dx] = eig(cov(xold'));
  %Q = sqrt(inv(Dx)) * Ex';
Q = mtimes(sqrt(inv(Dx)),Ex');

x = Q * xold;

 pC = (x*transpose(x))/m; %Pseudo covariance
% xSquare = diag(pC);
% C = (x*x')/m;
% cov(x')
% (x*x.')/size(x,2)
% cov(real(x(1,:)),imag(x(1,:)))
% cov(real(x(2,:)),imag(x(2,:)))

%x = x - mean(x,2)*ones(1,m);
%test1 = zeros(50,2);
%test2 = zeros(50,2);

% Condition in Theorem 1 should be < 0 when maximising and > 0 when 
% minimising E{G(|w^Hx|^2)}. 
% FIXED POINT ALGORITHM
 
  maxcounter = 100;
    W = eye(n);
    Wold = zeros(n);
    k=0;
    while (norm(abs(Wold'*W)-eye(n),'fro')>(n*1e-4))*(k<15*n)
         k = k+1;
         Wold = W;
         %%%%%%%%%%%%%
       for kk = 1:n
         yy = W(:,kk)'*x;
         p = 2;
       ga = getGgdMixtureGrads(yy,siggSq,U,mixCoef,p, 3);
       gb = getGgdMixtureGrads(yy,siggSq,U,mixCoef,p, 2);
       g =  getGgdMixtureGrads(yy,siggSq,U,mixCoef,p, 1);
       %%Put together
       temp1 =  ones(n,1)*conj(g);
       temp2 = temp1 .* x;
       gRad = mean(temp2,2);
       
       partOne = (mean(ga)*ones(n,1)).*W(:,kk);
       %% Use pseudocovariance
       if 1 == 1
        B = mean(conj(gb));
        %partTwo = (mean(B)*mean(x(kk,:).*(x(kk,:))))* conj(W(:,kk));
        partTwo = (B*pC) * conj(W(:,kk));
       else %do full average
           partTwo = 0; %Assume circular sources so partTwo is ~0
                        %For all but BPSK this is true, speeds up.
           B = (ones(n,1)*conj(gb)).*x*transpose(x)/m;
           partTwo = B*conj(W(:,kk));
       end
       W(:,kk) = -gRad + (partOne) + (partTwo); 
         %%%%%%%%

      end  %%Loop thru w's
      [E,D] = eig(W'*W);  %Orthogonalize
      W = W * E * inv(sqrt(D)) * E';
       
    end
    %%%%
    %clear EG;
      
%   absQAHW = abs((Q*A)'*W);
%   maximum = max(absQAHW);
%   SE = sum(absQAHW.^2) - maximum.^2 + (ones(1,n)-maximum).^2;
%   SSE = sum(SE);  

shat = W'*x;
numIter =1;% size(EG,2);

 Ahat = inv(Q)*W;

 k = k/n;
 
 
 %%%%%%%%%%%%%%%Internal Gradiant funcctions
 
function [val] = getGgdMixtureGrads(y,sigSq,mu,mixCoef,p, type)

if type == 0  %The cost not logged
    [val] = getGgdMixture(y,p,mixCoef,sigSq,mu);
elseif type == 1  %dw
    val = getGgdMixtureGrad(y,sigSq,mu,mixCoef,p);
elseif type == 2 %dwwdw = conj(gb)
    [val] =getGgdMixtureGradww(y,sigSq,mu,mixCoef,p);
elseif type == 3  %dww' = ga
    val = getGgdMixtureGradwcw(y,sigSq,mu,mixCoef,p);
end

%%%%Function w/o log
function [val] = getGgdMixture(y,p,mixCoef,sigSq,mu)

  N = size(mu,1);
  
  alphaV = gamma(4/p)*p/(sigSq*gamma(2/p)^2 *4*pi);
  betaV = (gamma(4/p)/(sigSq*gamma(2/p)*2)).^(p/2);
  sumVal = 0;
  for n = 1:N
       qw = (conj(y-mu(n,1)).*(y-mu(n,1))).^(p/2);
      temp = mixCoef(n,1)*alphaV * ...
          exp(-betaV*qw);
      sumVal = sumVal + temp;
  end
  
  val = (sumVal);


%%1st der
function [val] = getGgdMixtureGrad(y,sigSq,mu,mixCoef,p)
  N = size(mu,1);
  
  alphaV = gamma(4/p)*p/(sigSq*gamma(2/p)^2 *4*pi);
  betaV = (gamma(4/p)/(sigSq*gamma(2/p)*2)).^(p/2);
  G = getGgdMixture(y,p,mixCoef,sigSq,mu);
 
  sumVal = 0;
  for n = 1:N
      qw = conj(y-mu(n,1)).*(y-mu(n,1));
      fk = mixCoef(n,1)*alphaV * ...
          exp(-betaV*(qw).^(p/2));
      
      temp = fk.*(-betaV)*p/2 .* qw.^((p-2)/2) .* (y-mu(n,1));
      sumVal = sumVal + temp;
  end
  
  val = sumVal./G;

%%2nd dJ/ww
function [val] = getGgdMixtureGradww(y,sigSq,mu,mixCoef,p)
  N = size(mu,1);
  
  alphaV = gamma(4/p)*p/(sigSq*gamma(2/p)^2 *4*pi);
  betaV = (gamma(4/p)/(sigSq*gamma(2/p)*2)).^(p/2);
  G = getGgdMixture(y,p,mixCoef,sigSq,mu);

  Q = 0;temp2=0;
  for n = 1:N
      qw = conj(y-mu(n,1)).*(y-mu(n,1));
      fk = mixCoef(n,1)*alphaV * ...
          exp(-betaV*(qw).^(p/2));
      tempQ = -fk .*betaV*p/2 .*qw.^((p-2)/2).*(y-mu(n,1));
      Q = Q + tempQ;
      temp = -betaV*(tempQ *p/2.*qw.^((p-2)/2) .*(y-mu(n,1)) + ...
          fk.*((p^2-2*p)/4).*qw.^((p-4)/2) .* (y-mu(n,1)).^2);
      temp2 = temp2 + temp;
  end
  
  val = Q.^2 .*(-1./G.^2) + 1./G .* temp2;

%%2nd dJ/wwc
function [val] = getGgdMixtureGradwcw(y,sigSq,mu,mixCoef,p)
  N = size(mu,1);
  
  alphaV = gamma(4/p)*p/(sigSq*gamma(2/p)^2 *4*pi);
  betaV = (gamma(4/p)/(sigSq*gamma(2/p)*2)).^(p/2);
  G = getGgdMixture(y,p,mixCoef,sigSq,mu);

  Q = 0;temp2=0;Qc = 0;
  for n = 1:N
      qw = conj(y-mu(n,1)).*(y-mu(n,1));
      fk = mixCoef(n,1)*alphaV * ...
          exp(-betaV*(qw).^(p/2));
      tempQ = -fk .*betaV*p/2 .*qw.^((p-2)/2).*(y-mu(n,1));
      Q = Q + tempQ;
      tempQc = -fk .*betaV*p/2 .*qw.^((p-2)/2).*conj(y-mu(n,1));
      Qc = Qc + tempQc;
      temp = (tempQc .*qw.^((p-2)/2) .*(y-mu(n,1)) + ...
          fk.*(p/2.*qw.^((p-2)/2)));
      temp2 = temp2 + temp;
  end
  
  val = Q .*(-1./G.^2).*Qc + (-betaV)*(p/2)* 1./G .* temp2;

