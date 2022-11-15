%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "Joint Inference of Networks from Stationary Graph Signals"
% by Segarra, Wang, Uhler, and Marques
%
% Code for Fig. 1(b)
% Code by Santiago Segarra
%
% 11/25/17
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The goal of this code is to illustrate the joint recovery of social
% networks.

clear all;
close all;
rng(1111); % Fix random seed for replicability
addpath(['../'])
%% Load social networks and define the GSOs
% Load data
load('G_2_3');load('G_5_3');load('G_3_3'); 
% Define GSOs and make them symmetric and unweighted
S1 = G_2_3;S1 = S1 + S1';S1 = double(S1>0);
S2 = G_5_3;S2 = S2 + S2';S2 = double(S2>0);
S3 = G_3_3;S3 = S3 + S3';S3 = double(S3>0);

N = size(S1,1); % Get the number of nodes

%% Generate covariance matrices via a random polynomial mappings
L = 3; % Fix the size of the polynomial map (filters squared)
h1 = randn(L,1); % Draw the coefficients of the first polynomial
h2 = randn(L,1); % Draw the coefficients of the second polynomial
h3 = randn(L,1); % Draw the coefficients of the third polynomial

% Compute the covariances
H1 = zeros(N,N);H2 = zeros(N,N);H3 = zeros(N,N);
for ii = 1:L
    H1 = H1 + h1(ii)*S1^(ii-1);
    H2 = H2 + h2(ii)*S2^(ii-1);
    H3 = H3 + h3(ii)*S3^(ii-1);
end
C1 = H1^2;C2 = H2^2;C3 = H3^2;
    
% Check that the shifts and the covariances commute
if (max(max(abs(S1*C1 - C1*S1))) + max(max(abs(S2*C2 - C2*S2))) + max(max(abs(S3*C3 - C3*S3)))) < 10^-6
    disp('The GSOs and the true covariances commute ... CHECK')
end

%% Generate random signals from the computed covariances
max_signals = 100000;

X1 = H1*randn(N,max_signals);
X2 = H2*randn(N,max_signals);
X3 = H3*randn(N,max_signals);

%% For different numbers of signals solve the problems separately and jointly
signals_vec = [100 floor(100*sqrt(10)) 1000 floor(1000*sqrt(10)) 10000 floor(10000*sqrt(10)) 100000];

epsil_vec_1_sep = [2]; %100
epsil_vec_2_sep = [2]; %100
epsil_vec_3_sep = [10]; %100

%epsil_vec_1_sep = [0.166]; %1000
%epsil_vec_2_sep = [0.212]; %1000
%epsil_vec_3_sep = [3.089]; %1000

%epsil_vec_1_sep = [0.105]; %10000
%epsil_vec_2_sep = [0.107]; %10000
%epsil_vec_3_sep = [0.827]; %10000

error_sep = zeros(numel(signals_vec), 3);
error_joi = zeros(numel(signals_vec), 3);

alp = [1 1 1]';
bet = [10 10 10]';

for ss = 1:numel(signals_vec)
    sig_cons = signals_vec(ss);
    % Compute sample covariances
    C1_hat = cov(X1(:,1:sig_cons)');
    C2_hat = cov(X2(:,1:sig_cons)');
    C3_hat = cov(X3(:,1:sig_cons)');
    
    clear S1_hat_sep
    clear S2_hat_sep
    clear S3_hat_sep
    clear S1_hat_joi
    clear S2_hat_joi
    clear S3_hat_joi
    
   [S1_hat_sep, S2_hat_sep, S3_hat_sep, S1_hat_joi, S2_hat_joi, S3_hat_joi, last_epsil_valid_1, last_epsil_valid_2, last_epsil_valid_3] = ...
   infer_seperately_and_jointly(C1_hat, C2_hat, C3_hat, 0, 5, 0, 5, 0, 15, alp, bet);
    
%     % Solve the three graphs separately
%     warning('off', 'MATLAB:nargchk:deprecated') % Avoid bothersome warning
%     cvx_begin 
%     variable S1_hat_sep(N,N) symmetric;
%     minimize (norm(S1_hat_sep(:),1));
%     subject to
%     norm(C1_hat*S1_hat_sep - S1_hat_sep*C1_hat, 'fro') <= epsil_vec_1_sep(ss); % Numerical equality
%     S1_hat_sep >= 0; % Positivity constraints
%     diag(S1_hat_sep) <= 10^-6; % Numerical equality to 0
%     abs(sum(S1_hat_sep(:,1)) - 1) <= 10^-6; % Numerical equality to 1
%     cvx_end
%     
%     cvx_begin 
%     variable S2_hat_sep(N,N) symmetric;
%     minimize (norm(S2_hat_sep(:),1));
%     subject to
%     norm(C2_hat*S2_hat_sep - S2_hat_sep*C2_hat, 'fro') <= epsil_vec_2_sep(ss); % Numerical equality
%     S2_hat_sep >= 0; % Positivity constraints
%     diag(S2_hat_sep) <= 10^-6; % Numerical equality to 0
%     abs(sum(S2_hat_sep(:,1)) - 1) <= 10^-6; % Numerical equality to 1
%     cvx_end
% %     
%     cvx_begin 
%     variable S3_hat_sep(N,N) symmetric;
%     minimize (norm(S3_hat_sep(:),1));
%     subject to
%     norm(C3_hat*S3_hat_sep - S3_hat_sep*C3_hat, 'fro') <= epsil_vec_3_sep(ss); % Numerical equality
%     S3_hat_sep >= 0; % Positivity constraints
%     diag(S3_hat_sep) <= 10^-6; % Numerical equality to 0
%     abs(sum(S3_hat_sep(:,1)) - 1) <= 10^-6; % Numerical equality to 1
%     cvx_end
% %     
% %     % Solve them jointly
%     cvx_begin 
%     variable S1_hat_joi(N,N) symmetric;
%     variable S2_hat_joi(N,N) symmetric;
%     variable S3_hat_joi(N,N) symmetric;
%     minimize (alp(1)*norm(S1_hat_joi(:),1) + alp(2)*norm(S2_hat_joi(:),1) + alp(3)*norm(S3_hat_joi(:),1) + bet(1)*norm(S1_hat_joi(:)-S2_hat_joi(:),1) + bet(2)*norm(S1_hat_joi(:)-S3_hat_joi(:),1) + bet(3)*norm(S2_hat_joi(:)-S3_hat_joi(:),1));
%     subject to
%     norm(C1_hat*S1_hat_joi - S1_hat_joi*C1_hat, 'fro') <= 0.105; % Numerical equality
%     norm(C2_hat*S2_hat_joi - S2_hat_joi*C2_hat, 'fro') <= 0.107; % Numerical equality
%     norm(C3_hat*S3_hat_joi - S3_hat_joi*C3_hat, 'fro') <= 0.827; % Numerical equality
%     S1_hat_joi >= 0; % Positivity constraints
%     S2_hat_joi >= 0; % Positivity constraints
%     S3_hat_joi >= 0; % Positivity constraints
%     diag(S1_hat_joi) <= 10^-6; % Numerical equality to 0
%     diag(S2_hat_joi) <= 10^-6; % Numerical equality to 0
%     diag(S3_hat_joi) <= 10^-6; % Numerical equality to 0
%     abs(sum(S1_hat_joi(:,1)) - 1) <= 10^-6; % Numerical equality to 1
%     abs(sum(S2_hat_joi(:,1)) - 1) <= 10^-6; % Numerical equality to 1
%     abs(sum(S3_hat_joi(:,1)) - 1) <= 10^-6; % Numerical equality to 1
%     cvx_end
%     
%     % Compute errors by rescaling the true graphs to the recovered scale
      S1_resc = S1./sum(S1(:,1));
      S2_resc = S2./sum(S2(:,1));
      S3_resc = S3./sum(S3(:,1));
      error_sep(ss, :) = [norm(S1_hat_sep-S1_resc, 'fro')/norm(S1_resc, 'fro') norm(S2_hat_sep-S2_resc, 'fro')/norm(S2_resc, 'fro') norm(S3_hat_sep-S3_resc, 'fro')/norm(S3_resc, 'fro')];
      error_joi(ss, :) = [norm(S1_hat_joi-S1_resc, 'fro')/norm(S1_resc, 'fro') norm(S2_hat_joi-S2_resc, 'fro')/norm(S2_resc, 'fro') norm(S3_hat_joi-S3_resc, 'fro')/norm(S3_resc, 'fro')];
end

% 
%  figure;
%  subplot(3,3,1);imagesc(S1);
%  subplot(3,3,4);imagesc(S1_hat_sep);
%  subplot(3,3,2);imagesc(S2);
%  subplot(3,3,5);imagesc(S2_hat_sep);
%  subplot(3,3,3);imagesc(S3);
%  subplot(3,3,6);imagesc(S3_hat_sep);
%  subplot(3,3,7);imagesc(S1_hat_joi);
%  subplot(3,3,8);imagesc(S2_hat_joi);
%  subplot(3,3,9);imagesc(S3_hat_joi);
 

%% Plot the results

figure;
semilogx(signals_vec, error_joi(:, 1), '-bo')
hold all
semilogx(signals_vec, error_joi(:, 2), '-rs')
semilogx(signals_vec, error_joi(:, 3), '-gv')
semilogx(signals_vec, error_sep(:, 1), ':bo')
semilogx(signals_vec, error_sep(:, 2), ':rs')
semilogx(signals_vec, error_sep(:, 3), ':gv')
grid on
ylim([0 1])

xlabel('Nr. of signals observed','Interpreter','LaTex')
ylabel('Recovery error','Interpreter','LaTex')
legend({'Network 1 Joint', 'Network 2 Joint', 'Network 3 Joint', 'Network 1 Separate', 'Network 2 Separate', 'Network 3 Separate'})
set(gca, 'FontSize', 22)

%save('data_first_plot_social_2')



