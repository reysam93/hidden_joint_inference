%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "Joint Inference of Networks from Stationary Graph Signals"
% by Segarra, Wang, Uhler, and Marques
%
% Code for Fig. 1(c)
% Code by Santiago Segarra
%
% 11/25/17
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The goal of this code is to illustrate the joint recovery of social
% networks.

clear all;
close all;
rng(12345); % Fix random seed for replicability

%% Generate the three graphs from random choices of the students
N = 32;
S1 = zeros(N,N);S2 = zeros(N,N);S3 = zeros(N,N);
for i = 1:N
    S1(i,randperm(32,2)) = 1;
    S2(i,randperm(32,2)) = 1;
    S3(i,randperm(32,2)) = 1;
end
S1 = S1 + S1';S1 = double(S1>0);S1 = S1 - diag(diag(S1));
S2 = S2 + S2';S2 = double(S2>0);S2 = S2 - diag(diag(S2));
S3 = S3 + S3';S3 = double(S3>0);S3 = S3 - diag(diag(S3));


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
%signals_vec = [100 floor(100*sqrt(10)) 1000 floor(1000*sqrt(10)) 10000 floor(10000*sqrt(10)) 100000];
signals_vec = [100000];

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
   infer_seperately_and_jointly_2(C1_hat, C2_hat, C3_hat, 0, 15, 0, 15, 0, 15, alp, bet);
    

%     % Compute errors by rescaling the true graphs to the recovered scale
      S1_resc = S1./sum(S1(:,1));
      S2_resc = S2./sum(S2(:,1));
      S3_resc = S3./sum(S3(:,1));
      error_sep(ss, :) = [norm(S1_hat_sep-S1_resc, 'fro')/norm(S1_resc, 'fro') norm(S2_hat_sep-S2_resc, 'fro')/norm(S2_resc, 'fro') norm(S3_hat_sep-S3_resc, 'fro')/norm(S3_resc, 'fro')];
      error_joi(ss, :) = [norm(S1_hat_joi-S1_resc, 'fro')/norm(S1_resc, 'fro') norm(S2_hat_joi-S2_resc, 'fro')/norm(S2_resc, 'fro') norm(S3_hat_joi-S3_resc, 'fro')/norm(S3_resc, 'fro')];
end

% 
 figure;
 subplot(3,3,1);imagesc(S1);
 subplot(3,3,4);imagesc(S1_hat_sep);
 subplot(3,3,2);imagesc(S2);
 subplot(3,3,5);imagesc(S2_hat_sep);
 subplot(3,3,3);imagesc(S3);
 subplot(3,3,6);imagesc(S3_hat_sep);
 subplot(3,3,7);imagesc(S1_hat_joi);
 subplot(3,3,8);imagesc(S2_hat_joi);
 subplot(3,3,9);imagesc(S3_hat_joi);
 

%% Plot the results

error_sep = [
    1.0    0.9068    0.8914
    0.9533    0.9261    0.8341
    0.8582    0.9202    0.7638
    0.8597    0.8433    0.7261
    0.8120    0.7186    0.4806
    0.8565    0.6940    0.3361
    0.7468    0.4285    0.1689
    ];

error_joi = [

    1.0    0.9034    0.8939
    0.9571    0.9224    0.8387
    0.9165    0.9161    0.7685
    0.8974    0.8448    0.7675
    0.8540    0.7850    0.5823
    0.8842    0.7643    0.4320
    0.8433    0.5039    0.2440
    ];

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

save('data_second_plot_social_2')



