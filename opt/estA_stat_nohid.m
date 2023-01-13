function [A] = estA_stat_nohid(C,regs,max_iters,verb)
if nargin < 3
    max_iters = 5;
end
if nargin < 4
    verb = false;
end

N = size(C,1);
K = size(C,3);
alpha = regs.alpha;
beta = regs.beta;
mu = regs.mu;
del1 = regs.delta1;

% disp('Starting iterative algorithm')
Ao_prev = ones(N,N,K);
f0_prev = 1e6;
for i=1:max_iters
    W_Ao = alpha*ones(N,N,K)./(Ao_prev+del1);

    % Step 1: infer Ao and Aoh
    cvx_begin quiet
        variable A(N,N,K) symmetric nonnegative
        % Sparsity of Ao
        f0 = vec(W_Ao)'*vec(A);
        for k=1:K
            % Commutativity penalty
            f0 = f0 + mu*norm(C(:,:,k)*A(:,:,k) - A(:,:,k)*C(:,:,k),'fro');

            % Graph similarity penalty
            for j=1:(k-1)
               f0 = f0 + beta*norm(vec(A(:,:,k)-A(:,:,j)),1);
            end
        end

        minimize(f0)
        subject to
            for k=1:K
                diag(A(:,:,k)) == 0;
                sum(A(:,:,k))>= 1;
            end
    cvx_end

    norm_Ao_prev = norm(vec(Ao_prev),2)^2;
    diff_Ao = norm(vec(A-Ao_prev),2)^2/norm_Ao_prev;
    f0_diff = abs(f0 - f0_prev);
    Ao_prev = A;
    
    comm = 0;
    
    if verb
        disp(['Iter: ' num2str(i) ' status: ' cvx_status  '   f0-f0_prev: ' num2str(f0_diff)...
            '   Ao-Ao_prev: ' num2str(diff_Ao)...
            '   Comm: ' num2str(comm)])
    end
    
    % Stop condition
    if f0_diff < 1e-2
        if verb
            disp('Convergence achieved!')
        end
        break
    end
    f0_prev = f0;
end