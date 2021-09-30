function S_ggl = gglasso(C,regs)
lambda1 = regs.lambda1;
lambda2 = regs.lambda2;
eps_thresh = 1e-6;

K = size(C,3);
N = size(C,1);

cvx_begin quiet
    variable Theta(N,N,K) symmetric nonnegative

    f0 = 0;
    for k=1:K
        % Negative log-likelihood
        f0 = f0 - log_det(Theta(:,:,k)) + trace(Theta(:,:,k)*C(:,:,k));
        
        % Sparse penalties
        f0 = f0 + lambda1*norm((1-eye(N)).*Theta(:,:,k),1);

%         % Graph similarity penalties (FGL)
%         for j=1:(k-1)
%            f0 = f0 + lmbda2*norm((1-eye(N)).*(Theta(:,:,k)-Theta(:,:,j)),1);
%         end
    end
    % Graph similarity penalties (GGL)
    f0 = f0 + lambda2*sum(sum(norms(Theta,2,3).*(1-eye(N))));

    minimize(f0)
cvx_end

S_ggl = zeros(N,N,K);
for k=1:K
    S_ggl(:,:,k) = Theta(:,:,k).*(1-eye(N));
    
    first_col = 1;
    while first_col<=N && sum(S_ggl(:,first_col,k))==0
        first_col = first_col + 1;
    end
    if first_col<=N
        S_ggl(:,:,k) = S_ggl(:,:,k)/sum(S_ggl(:,first_col,k));
    end
end

end

