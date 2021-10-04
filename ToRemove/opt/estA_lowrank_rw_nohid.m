function [A,P] = estA_lowrank_rw_nohid(C, regs)
A = [];
P = [];

N = size(C,1);
K = size(C,3);

alpha = regs.alpha;%1e-2
beta = regs.beta;%1e-1
mu = regs.mu;%100
max_iters = regs.max_iters;
del = regs.delta1;

A_prev = ones(N,N,K);

for i=1:max_iters
    W_A = alpha*ones(N,N,K)./(A_prev+del);
    
    cvx_begin quiet
        variable A(N,N,K) symmetric nonnegative
        f0 = vec(W_A)'*vec(A);
        for k=1:K
            f0 = f0 + mu*norm(C(:,:,k)*A(:,:,k)-A(:,:,k)*C(:,:,k),'fro');
            for j=1:(k-1)
               f0 = f0 + beta*norm(vec(A(:,:,k)-A(:,:,j)),1);
            end
        end

        minimize(f0)
        subject to
            for k=1:K
                diag(A(:,:,k)) == 0;
                sum(A(:,1,k))== 1;
            end
    cvx_end
    A_prev = A;
end