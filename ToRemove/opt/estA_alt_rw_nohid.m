function [A] = estA_alt_rw_nohid(C,regs,max_iters,verb)
if nargin < 3
    max_iters = 10;
end
if nargin < 4
    verb = false;
end

N = size(C,1);
K = size(C,3);

alpha = regs.alpha;
beta = regs.beta;
mu = regs.mu;
del = regs.delta1;

if verb
    disp('Starting iterative algorithm')
end

norm_C = norm(vec(C),2)^2;
A_prev = ones(N,N,K);
for i=1:max_iters
    % Step 1: infer Ao and Aoh    
    W_A = alpha*ones(N,N,K)./(A_prev+del);
    cvx_begin quiet
        variable A(N,N,K) symmetric nonnegative
        % Reweighted Sparse penalties
        f1 = vec(W_A)'*vec(A);
        for k=1:K
            % Commutativity penalty
            f1 = f1 + mu*norm(C(:,:,k)*A(:,:,k)-A(:,:,k)*C(:,:,k),'fro');
            % Graph similarity penalties
            for j=1:(k-1)
               f1 = f1 + beta*norm(vec(A(:,:,k)-A(:,:,j)),1);
            end
        end

        minimize(f1)
        subject to
            for k=1:K
                diag(A(:,:,k)) == 0;
                sum(A(:,1,k))== 1;
            end
    cvx_end
    
    norm_A = norm(vec(A),2)^2;
    diff_A = norm(vec(A-A_prev),2)^2/norm_A;
    A_prev = A;
    
    comm = norm(vec(pagemtimes(C,A)-pagemtimes(A,C)),2)^2;
    comm_norm = comm/norm_A/norm_C;
    if verb
        disp(['Iter: ' num2str(i) '   Ao-Ao_prev: ' num2str(diff_A)...
            '   Comm: ' num2str(comm) '   Comm norm: ' num2str(comm_norm)])
    end
    
    % Stop condition
    if diff_A < 1e-3
        if verb
            disp('Convergence achieved!')
        end
        break
    end
end