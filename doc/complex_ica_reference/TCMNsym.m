%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reference
%Mike Novey and T. Adali, "Complex ICA by Negentropy Maximization" in
% IEEE Journal on Neural Networks. 2008, vol. 19, pp. 596--609, April 2008
%
% Function to perform ICA using T-CMN algorithm 
% where typeStr is the function type, i.e., G(y) = cosh(y)
% If typeSTr = 'pow' then inVal is the exponent,i.e., G(y) = y^inVAl
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
 
 
%%Symetric orthogonalization
function [Ahat, shat, k] = doCMNsym(xold,typeStr,inVal)
global globalAlpha;
 type = 0;
 localAlpha = 1;
insData = 0;
 if strcmp(typeStr,'def')==1
     type =0;
 elseif strcmp(typeStr,'atanh')==1
     type = 1;
 elseif strcmp(typeStr,'asinh')==1
     type = 2;
 elseif strcmp(typeStr,'tanh')==1
     type = 3;
  elseif strcmp(typeStr,'cosh')==1
     type = 4;
  elseif strcmp(typeStr,'acosh')==1
     type = 5;
 elseif strcmp(typeStr,'x^2')==1
     type = 6;
 elseif strcmp(typeStr,'asin')==1
     type = 7;
  elseif strcmp(typeStr,'tan')==1
     type = 8;
 elseif strcmp(typeStr,'poly')==1
     type = 9;
  elseif strcmp(typeStr,'cos')==1
     type = 10;
 elseif strcmp(typeStr,'asinhasin')==1
       type = 11;
 elseif strcmp(typeStr,'coshcos')==1
     type = 12;
 elseif strcmp(typeStr,'pow')==1
     type = 13;
     localAlpha = inVal; %Exponent
  elseif strcmp(typeStr,'fdist')==1
     type = 14;
     globalAlpha = inVal; %Exponent
 else
     disp('error in function type');
     return;
 end
 
 
 tol = 1e-5;
 maxcounter = 15;
  [n,m] = size(xold);
 
  % Whitening of s:
[Ex, Dx] = eig(cov(xold'));
  %Q = sqrt(inv(Dx)) * Ex';
Q = mtimes(sqrt(inv(Dx)),Ex');
x = Q * xold;
 
 %Pseudocovariance
 pC = (x*transpose(x))/m;
 
 
% FIXED POINT ALGORITHM Symmetric
  %W = zeros(n,n) + j*zeros(n,n);
  W = eye(n);
  Wold = zeros(n);
  k=0;
 
   while (norm(abs(Wold'*W)-eye(n),'fro')>(n*tol) && k < maxcounter*n)
         k = k+1;
         Wold = W;
     for kk=1:n %Loop thru sources
      mu = 4;
      alpha = 1.25;
      yy = W(:,kk)'*x;
       
      %%Fixed point
      if   type == 13     %POW, i.e., Z^localAlpha
          absy = abs(yy).^2;
          %Prevent singularities in derivative
          if localAlpha < 1
              a2 = .001;
          else
              a2 =0;
          end
          u = (absy + a2);
          u1 = localAlpha*u.^(localAlpha-1);
          u2 = localAlpha*(localAlpha-1)*u.^(localAlpha-2);
          gRad = mean(ones(n,1)*(u1.*conj(yy)).*x,2);
          ggg = mean(u2.*absy + u1);
          B = mean(u2.* (conj(yy).^2))*pC;
          W(:,kk)  =  Wold(:,kk)*ggg -(gRad) + (B*conj(Wold(:,kk))); 
          % w =  wold -mu*(gRad); 
          %mu = mu*.96;
      else  %All other functions
       absy = abs(yy).^2;
       G =  getG(type,yy);
       g =  getGdir(type,yy);
       gp = getGsecDir(type,yy);
       gRad = mean(ones(n,1)*(g.*conj(G)).*x,2);
       ggg = mean(g.*conj(g)); 
       B = mean(conj(G).* gp)*pC;
       W(:,kk)  =  Wold(:,kk)*ggg -(gRad) + (B*conj(Wold(:,kk))); 
      end
         
     end %Loop thru sources
    [E,D] = eig(W'*W);
    W = W * E * inv(sqrt(D)) * E';
  end; %While
 
 shat = W'*x;
 
 Ahat = inv(Q)*W;
k = k/n;
%  w1 = (W(:,2));
%   N = size(x,2);
% p = localAlpha;
% plot(abs(shat(2,:)));grid on;
% plot(abs(shat(1,:)));grid on;
% wp = x(:,512);
% w1'*x*x'*w1/(N)
% cc = w1'*wp*wp'*w1/N
% 1-cc
%  aa = mean((abs(w1'*x).^2).^(p-1));
%  Te = aa * x*x'/N
%  A = Te*w1 ;
% gg = A/(w1'*A) - w1
% 
%  T = ((ones(2,1)*(((w1'*x).*conj(w1'*x)).^(p-1)).*x)*x')/N
%  A = T*w1 ;
% gg = A/(w1'*A) - w1
% hold off;plot(abs(shat(2,:)));grid on;
% hold on;plot(abs(shat(1,:))-5,'r');grid on;
% title(num2str(norm(gg)));
%  x(:,512) = x(:,500);
%  
% x*x'/N + wp*wp'/N
% w1'*x*x'*w1/(N) + w1'*wp*wp'*w1/N
%  aa = (((w1'*x*x'*w1)^(p-1)))/N
%  aa = mean((abs(w1'*x).^2).^(p-1))
%  bb = [mean(conj(w1'*x).*x(1,:)) mean(conj(w1'*x).*x(2,:))].'
%  bbb=  aa*bb
%  bbb = bbb/norm(bbb)
%  cc1= mean(((w1'*x).*conj(w1'*x)).^(p-1) .*conj(w1'*x).*x(1,:))
%  cc2= mean(((w1'*x).*conj(w1'*x)).^(p-1) .*conj(w1'*x).*x(2,:))
%  cccc = [cc1 cc2].';
%  cccc = cccc/norm(cccc)
% norm(w1-cccc)
% norm(w1 - bbb)
% norm(cccc-bbb)
%  T = ((ones(2,1)*(((w1'*x).*conj(w1'*x)).^(p-1)).*x)*x')/N
%  Te = aa * x*x'/N
% A = T*w1 + inv(N)*(w1'*wp*wp'*w1)^(p-1)*wp*conj(wp'*w1);
% gg = A/(w1'*A) - w1
% 
% A = Te*w1 + inv(N)*(w1'*wp*wp'*w1)^(p-1)*wp*conj(wp'*w1);
% gg = A/(w1'*A) - w1
% insData = norm(T*w1 - w1'*T*w1*w1);
% title(num2str(insData));
 %%Internal functions
 
 function val = getG(type,yy)
     global globalAlpha;
        val = 0;
        if type == 0  %log
            yy =  abs(yy).^2;
           val = log(eps + yy);
        elseif type ==1  %atanh
            val = atanh(yy);
        elseif type ==2  %asinh
            val = asinh(yy);
        elseif type ==3  %tanhh
            val = tanh(yy);
        elseif type ==4  %cosh
            val = cosh(yy);
        elseif type ==5  %acosh
            val = acosh(yy);
        elseif type ==6  %kurrt
            val = yy.^2;
            %val = conj(conj(conj(yy).*yy).*yy).*yy;
        elseif type ==7  %asin
            val = asin(yy);
         elseif type ==8  %exp
            val = tan(yy);
         elseif type ==9  %Asin MC
            val = yy + Kin*yy.^3 ;
           % val = yy.^.25;
         elseif type ==10  %exp
            val = cos(yy);
         elseif type ==11  %asinhasin
            val = asinh(yy)+asin(yy);
         elseif type ==12 %coshcos
            val = cosh(yy) + cos(yy);
        elseif type == 14 %F
            val = fpdf(yy,globalAlpha,globalAlpha*2);
        end
        
      
%%Derivitives  
 function val = getGdir(type,yy)
     global globalAlpha;
        val = 0;
         
        if type == 0
            yy =  abs(yy).^2;
           val = 1./(eps + yy);
        elseif type ==1 %atanh
            val = 1./(1 - yy.^2);
        elseif type ==2 %asinh
            val = 1./sqrt(ones(size(yy)) + yy.^2);
        elseif type ==3 %tanh
            val = sech(yy).^2;
        elseif type ==4 %cosh
            val = .5*(exp(yy)-exp(-yy));
         elseif type ==5 %acosh
            val = 1./(sqrt(yy-ones(size(yy))).*sqrt(yy+ones(size(yy))));
         elseif type ==6 %kurt
             val = 2*yy;
             %val = 4*conj(conj(yy).*yy).*yy;
         elseif type ==7 %asin
             val = 1./sqrt(1-yy.^2);
         elseif type == 8 %exp
             val = 1+tan(yy).^2;
         elseif type == 9 %  asinH Mc
             val = 1+3*Kin*yy.^2;
             %val = .25*yy.^(-.75);
         elseif type == 10 %cos
             val = -sin(yy);
         elseif type ==11 %asinhasin
            val = (1./sqrt(ones(size(yy)) + yy.^2)) + ( 1./sqrt(1-yy.^2));
         elseif type ==12 %coshcos
            val = sinh(yy) - sin(yy);
         elseif type == 14 %F
             delt = .01;
            val1 = fpdf(yy+delt,globalAlpha,globalAlpha*2);
            val2 = fpdf(yy-delt,globalAlpha,globalAlpha*2);
            val = (val1-val2)/delt;
        end
        
 
%%2nd Derivitives  
 
function val = getGsecDir(type,yy)
    global globalAlpha;
        val = 0;
        
        if type == 0
            yy =  abs(yy).^2;
           val = -1./(eps + yy).^2;
        elseif type ==1 %atanh
            val = (2*yy)./(ones(size(yy)) - 2*yy.^2 + yy.^4);
        elseif type ==2 %asinh
            val = -yy./(ones(size(yy)) + yy.^2).^(3/2);
        elseif type ==3 %tanh
            temp = (8*exp(2*yy))/((ones(size(yy))+exp(2*yy)).^3);
            val = temp.*(ones(size(yy)) - exp(2*yy));
         elseif type ==4 %cosh
            val = cosh(yy);
         elseif type ==5 %acosh
           val = (-.5./((yy-ones(size(yy))).*yy+ones(size(yy)))).*((yy-ones(size(yy))).^(-.5) .*(yy+ones(size(yy))).^.5 ...
                + (yy+ones(size(yy))).^(-.5).*(yy-ones(size(yy))).^.5);
         elseif type == 6 %kurt
             val = 2*ones(size(yy));
         elseif type ==7 %asin
             val = .5*(yy)./(1-yy.^2).^1.5;
        elseif type == 8 %rxp
            val = 2*tan(yy).*(1+tan(yy).^2); 
        elseif type == 9 %asinHMc
          val = 6*Kin*yy;   
          %val = -.1875*yy.^(-1.75);
        elseif type == 10 %cos
          val = -cos(yy);    
        elseif type ==11 %asinhasin
            val = (-yy./(ones(size(yy)) + yy.^2).^(3/2)) + ( .5*(yy)./(1-yy.^2).^1.5);
        elseif type ==12 %coshcos
            val = cosh(yy) -cos(yy) ;
         elseif type == 14 %F
             delt = .01;
            val1 = getGdir(type,yy+delt);
            val2 = getGdir(type,yy-delt);
            val = (val1-val2)/delt;        
        end
      
   
