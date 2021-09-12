function S_pd = pair_diff(X)
    K = length(X);
    N = size(X{1},1);
    RK = size(X{1},2);
    L = nchoosek(N,2);
    
    %----------------------------------------
    % Manipulating parameters:
    
    [low_tri_1,low_tri_2] = ind2sub([N,N],find(tril(ones(N,N))-eye(N)));
    [upp_tri_1,upp_tri_2] = ind2sub([N,N],find(triu(ones(N,N))-eye(N)));

    U_ind = (low_tri_1-1)*N + low_tri_2;
    U_ind = kron(ones(K,1),U_ind) + kron(([0:K-1]')*(N^2),ones(L,1));

    L_ind = (low_tri_2-1)*N + low_tri_1;
    L_ind = kron(ones(K,1),L_ind) + kron(([0:K-1]')*(N^2),ones(L,1));
    
    [low_tri_K_1,low_tri_K_2] = ind2sub([K,K],find(tril(ones(K,K))-eye(K)));
    D = zeros(K*(K-1)/2,K);
    if K>2
        for l=1:size(D,1)
            D(l,low_tri_K_1(l)) =  1;
            D(l,low_tri_K_2(l)) = -1;
        end
    else
        D(1,1) = -1;
        D(1,2) =  1;
    end
    W = kron(D,eye(L));
    
    %----------------------------------------
    % Estimated covariances:
    C_est = cell(K,1);
    for k=1:K
        C_est{k} = cov(X{k}');
    end
    Sigma = zeros(K*N*N,K*N*N);
    for k=1:K
        diag_block = zeros(K,K);
        diag_block(k,k)=1;
        
        Sigma = Sigma + ...
            kron( diag_block,...
                  -kron(C_est{k},eye(N)) + kron(eye(N),C_est{k}) );
    end
    M = Sigma(:,L_ind)+Sigma(:,U_ind);
    
    %----------------------------------------
    % Initialize:
    
    %-------------------
    % s-initialize
    s_est = binornd(1,.5,K*L,1);
    
    %-------------------
    % p-initialize
    p_est = s_est;
    
    %-------------------
    % l-initialize
    l_est = W*s_est;
    
    %-------------------
    % u-initialize
    u1 = zeros(L*nchoosek(K,2),1);
    u2 = zeros(K*L,1);
    
    %----------------------------------------
    % Choose parameters:
    
    rho1 = 10; rho2 = 10; theta = 10;
    eps_abs = 0; eps_rel = 1e-3;
    
    Phi_s1 = theta*(M'*M) + rho1*(W'*W) + rho2*eye(K*L);
    
    %----------------------------------------
    % Estimation:
    
    pr = cell(1e3,1);
    dr = cell(1e3,1);
    eps_pri = cell(1e3,1);
    eps_dua = cell(1e3,1);
    
    for admm_iter=1:1e3
        %-------------------
        % s-update
        Phi_s2 = rho1*(W'*(l_est-u1)) + rho2*(p_est-u2);
        s_est = Phi_s1\Phi_s2;
        
        %-------------------
        % p-update
        p_last = p_est;
        for k=1:K
            ret = s_est + u2;
            ret = ret((k-1)*L+1:k*L);
            p_est((k-1)*L+1:k*L) = (ret-min(ret))/max(ret-min(ret));
        end
        p_est(p_est>=.5)=1;
        p_est(p_est <.5)=0;
        
        %-------------------
        % l-update
        l_last = l_est;
        z = W*s_est + u1;
        l_est = (z-(1/rho1)).*(z-(1/rho1) > 0) - (-z-(1/rho1)).*(-z-(1/rho1) > 0);
        
        %-------------------
        % u-updates
        u1 = u1 + W*s_est - l_est;
        u2 = u2 + s_est - p_est;
        
        %-------------------
        pr{admm_iter} = norm(W*s_est - l_est) + norm(s_est - p_est);
        dr{admm_iter} = norm(rho1*(W'*(l_est-l_last))) + norm(rho2*(p_est-p_last));
        eps_pri{admm_iter} = eps_abs*sqrt(length(s_est)) + eps_rel*max(norm(W*s_est),norm(l_est)) + ...
                             eps_abs*sqrt(length(s_est)) + eps_rel*max(norm(s_est),norm(p_est));
        eps_dua{admm_iter} = eps_abs*sqrt(length(u1)) + eps_rel*norm(rho1*(W'*u1)) + ...
                             eps_abs*sqrt(length(u2)) + eps_rel*norm(rho2*u2);
        
        if ((sum(cell2mat(pr)<=cell2mat(eps_pri))>0) && (sum(cell2mat(dr)<=cell2mat(eps_dua))>0))
            break
        end
        
        % End estimation
        %----------------------------------------
    end
    
    S_pd = cell(K,1);
    low_tri_ind = find(tril(ones(N,N))-eye(N));
    for k=1:K
        S_pd{k} = zeros(N,N);
        S_pd{k}(low_tri_ind) = p_est((k-1)*L+1:k*L);
        S_pd{k} = S_pd{k} + S_pd{k}';
    end
    
end

