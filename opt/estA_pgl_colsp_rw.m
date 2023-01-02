function [Ao,P] = estA_pgl_colsp_rw(Co,regs,max_iters,verb)
if nargin < 3
    max_iters = 5;
end
if nargin < 4
    verb = false;
end

O = size(Co,1);
K = size(Co,3);
alpha = regs.alpha;
beta = regs.beta;
eta = regs.eta;
gamma = regs.gamma;
mu = regs.mu;
del1 = regs.delta1;

% disp('Starting iterative algorithm')
Ao_prev = ones(O,O,K);
f0_prev = 1e6;
for i=1:max_iters
    W_Ao = alpha*ones(O,O,K)./(Ao_prev+del1);

    % Step 1: infer Ao and Aoh
    cvx_begin quiet
        variable Ao(O,O,K) symmetric nonnegative
        variable P(O,O,K)

        f0 = vec(W_Ao)'*vec(Ao);
        for k=1:K
            % Sparse penalties
            f0 = f0 + gamma*sum(norms(P(:,:,k)));
            % Commutativity penalty
            f0 = f0 + mu*norm(Co(:,:,k)*Ao(:,:,k)+P(:,:,k)...
                -Ao(:,:,k)*Co(:,:,k)-P(:,:,k)','fro');
            % Graph similarity penalties
            for j=1:(k-1)
               f0 = f0 + beta*norm(vec(Ao(:,:,k)-Ao(:,:,j)),1) +...
                   eta*sum(norms([P(:,:,k); P(:,:,j)],2));
            end
        end

        minimize(f0)
        subject to
            for k=1:K
                diag(Ao(:,:,k)) == 0;
                sum(Ao(:,:,k))>= 1;
            end
    cvx_end

    norm_Ao_prev = norm(vec(Ao_prev),2)^2;
    diff_Ao = norm(vec(Ao-Ao_prev),2)^2/norm_Ao_prev;
    f0_diff = abs(f0 - f0_prev);
    Ao_prev = Ao;
    
%     comm = norm(Co(:,:,k)*Ao(:,:,k)+P(:,:,k)...
%                 -Ao(:,:,k)*Co(:,:,k)-P(:,:,k)','fro');
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