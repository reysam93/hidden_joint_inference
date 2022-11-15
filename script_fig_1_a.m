%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "Joint Inference of Networks from Stationary Graph Signals"
% by Segarra, Wang, Uhler, and Marques
%
% Code for Fig. 1(a)
% Code by Santiago Segarra
%
% 11/25/17
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The goal of this code is to illustrate the theoretical result in Theorem
% 1.

close all
clear all
rng(1234); % Fix random seed for replicability


tot_runs = 500; % Number of realizations of the recovery problem

rec_vec = zeros(tot_runs, 1); % Store binary recovery
gamm_vec = zeros(tot_runs, 1); % Store associated gamma
cvx_status_vec = cell(tot_runs, 1);
cond_1_vec = zeros(tot_runs, 1);

% IMPORTANT: We should never observe negative recovery with gamma < 1.
% That would contradict the theorem.

for tt = 1:tot_runs

    fprintf('Running realization number: %d.0 \n', tt)
    %% Draw a pair of ER random graphs
    K = 2; % Nr of graphs being considered
    N = 20; % Nr of nodes in each graph
    p = 0.1; % Probability of edge appearance
    S1 = generate_connected_ER(N, p); % We find graph 1 randomly
    rewire = 3;
    
    flag_connected_S2 = 0;
    
    while flag_connected_S2 == 0
        % We find S2 by rewiring 3 edges from S1
        edge_positions = find(vec(triu(S1)));
        edges_to_delete = edge_positions(randperm(size(edge_positions,1), rewire));
        matrix_to_delete = zeros(N,N); % Generate a matrix to substract from S1
        matrix_to_delete(edges_to_delete) = 1; % Put ones in the edges to delete
        matrix_to_delete = matrix_to_delete + matrix_to_delete'; % Make it undirected
        S2 = S1 - matrix_to_delete;
        do_rewiring = 0; % Do the necessary rewirings
        while do_rewiring < rewire
            i = randi([1 N-1]);
            j = randi([i+1 N]);
            if S2(i,j) == 0 % Make sure that the edge to add is not already present
                S2(i,j) = 1;
                S2(j,i) = 1;
                do_rewiring = do_rewiring + 1;
            end
        end
        % Check if S2 is connected
        Lapl = diag(sum(S2)) - S2;
        ll = eig(Lapl);
        if size(find(ll<=10^-6))==1
            flag_connected_S2 = 1;
        end
    end
    
    %% Generate covariance matrices via a random polynomial mappings
    L = 5; % Fix the size of the polynomial map (filters squared)
    h1 = randn(L,1); % Draw the coefficients of the first polynomial
    h2 = randn(L,1); % Draw the coefficients of the second polynomial
    
    % Compute the covariances
    H1 = zeros(N,N);
    H2 = zeros(N,N);
    for ii = 1:L
        H1 = H1 + h1(ii)*S1^(ii-1);
        H2 = H2 + h2(ii)*S2^(ii-1);
    end
    
    C1 = H1^2;
    C2 = H2^2;
    
    % Check that the shifts and the covariances commute
    if (max(max(abs(S1*C1 - C1*S1))) + max(max(abs(S2*C2 - C2*S2)))) < 10^-6
        disp('The GSOs and the true covariances commute ... CHECK')
    end
    
    %% Solve the relaxed optimization problem
    
    alp = [1 1]'; % Define the vector of alphas
    bet = 1; % Define the vector of betas
    
    clear S1_hat
    clear S2_hat
    
    %
    disp('Solving the convex optimization problem ...')
    warning('off', 'MATLAB:nargchk:deprecated') % Avoid bothersome warning
    cvx_begin quiet
    variable S1_hat(N,N) symmetric;
    variable S2_hat(N,N) symmetric;
    minimize (alp(1)*norm(S1_hat(:),1) + alp(2)*norm(S2_hat(:),1) + bet* norm(S1_hat(:) - S2_hat(:), 1));
    subject to
    norm(C1*S1_hat - S1_hat*C1, 'fro') <= 10^-8; % Numerical equality
    norm(C2*S2_hat - S2_hat*C2, 'fro') <= 10^-8; % Numerical equality
    S1_hat >= 0; % Positivity constraints
    S2_hat >= 0; % Positivity constraints
    diag(S1_hat) <= 10^-8; % Numerical equality to 0
    diag(S2_hat) <= 10^-8; % Numerical equality to 0
    abs(sum(S1_hat(:,1)) - 1) <= 10^-8; % Numerical equality to 1
    abs(sum(S2_hat(:,1)) - 1) <= 10^-8; % Numerical equality to 1
    cvx_end
    
    cvx_status_vec(tt) = cellstr(cvx_status);
    
    % Rescale the obtained graphs to the true scale for comparison
    S1_hat = S1_hat*sum(S1(:,1));
    S2_hat = S2_hat*sum(S2(:,1));
    
    % Plot the true and recovered graphs
%     figure;
%     subplot(2,2,1);imagesc(S1);
%     subplot(2,2,2);imagesc(S2);
%     subplot(2,2,3);imagesc(S1_hat);
%     subplot(2,2,4);imagesc(S2_hat);
    disp('DONE solving the convex optimization problem')
    
    % Check if the recovered GSOs are the true ones
    if (max(max(abs(S1 - S1_hat))) + max(max(abs(S2 - S2_hat)))) < 10^-3
        rec_vec(tt) = 1;
        disp('The GSOs recovered coincide with the true ones!')
    else
        rec_vec(tt) = 0;
        disp('The GSOs recovered DO NOT coincide with the true ones!')
    end
    %}
    %% Check the theoretical guarantees
    % Here I use the same notation as in the paper
    % Build Psi
    Z = [1 -1];
    Psi = kron([diag(alp);bet*Z], eye(N^2));
    % Build Phi
    cal_D = 1:N+1:N^2;
    C_x_1 = - kron(C1, eye(N)) + kron(eye(N), C1);
    C_x_2 = - kron(C2, eye(N)) + kron(eye(N), C2);
    C_x = [C_x_1 zeros(N^2, N^2); zeros(N^2, N^2) C_x_2];
    B = zeros(nchoosek(N, 2), N^2);
    indic = 0;
    for i = 1:N-1
        for j = i+1:N
            indic = indic + 1;
            B_row_aux = zeros(N,N);
            B_row_aux(i,j) = 1;
            B_row_aux(j,i) = -1;
            B(indic, :) = B_row_aux(:)';
        end
    end
    I_N2 = eye(N^2);
    e1 = zeros(N,1);
    e1(1) = 1;
    Phi_1 = kron(eye(K), B);
    Phi_2 = kron(eye(K), I_N2(cal_D, :));
    Phi_3 = C_x;
    Phi_4 = kron(eye(K), kron(e1, ones(N,1))');
    Phi = [Phi_1;Phi_2;Phi_3;Phi_4];
    s_star = [S1(:)./sum(S1(:,1));S2(:)./sum(S2(:,1))];
    cal_J = find(s_star);
    cal_L = find(Psi*s_star);
    cal_L_c = setdiff(1:size(Psi*s_star,1), cal_L);
    
    % Check condition 1
    M = Phi(:, cal_J);
    if rank(M) == size(M, 2)
        cond_1_vec(tt) = 1;
        disp('Condition 1 of the theorem is satisfied!')
    else
        cond_1_vec(tt) = 0;
        disp('Condition 1 of the theorem is NOT satisfied!')
    end
    
    del = 10^-3;
    M2 = Psi(cal_L_c, :) * inv(del^-2*Phi'*Phi + Psi(cal_L_c, :)'*Psi(cal_L_c, :))*Psi(cal_L, :)';
    gamm_vec(tt) = norm(M2,inf);
    
     if rec_vec(tt) == 0 && gamm_vec(tt)<=0.9999
         disp('PASO ALGO RARO')
     end
    
end

%% Plot the output figure
gam_successes = gamm_vec(rec_vec==1);
gam_failures = gamm_vec(rec_vec==0);

hist_suc = histogram(gam_successes, 0:0.25:max(gamm_vec));
hist_suc_values = hist_suc.Values;
hist_failures = histogram(gam_failures, 0:0.25:max(gamm_vec));
hist_failures_values = hist_failures.Values;

bar(0.125:0.25:max(gamm_vec)-0.125, [hist_suc_values' hist_failures_values'], 1, 'Stacked');

save('data_hist_theorem_2');
grid on

xlim([0 6])
xlabel('$\gamma$ as defined (5)','Interpreter','LaTex')
ylabel('Nr. of experiments','Interpreter','LaTex')
legend({'Successful recovery', 'Failed recovery'})
set(gca, 'FontSize', 22)
