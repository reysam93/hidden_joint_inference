function [Ao,P] = estA_grouplasso_rw(Co, regs)
Ao = [];
P = [];
verb = false;
O = size(Co,1);
K = size(Co,3);

%Cpoly
% alpha = regs.alpha;
% gamma = regs.gamma;
% beta = regs.beta;
% eta = regs.eta;
% mu = regs.mu;

%Cmrf
alpha = 1e-2;%regs.alpha;
gamma = 1e2;%regs.gamma;
beta = 0;%regs.beta;
eta = 0;%regs.eta;
mu = 1e4;%regs.mu;

max_iters = regs.max_iters;
del1 = regs.delta1;

Ao_prev = ones(O,O,K);

for i=1:max_iters
    W_Ao = alpha*ones(O,O,K)./(Ao_prev+del1);
    
    cvx_begin quiet
        variable Ao(O,O,K) symmetric nonnegative
        variable P(O,O,K)
        variable Pm(O,O,K) nonnegative
        variable Pp(O,O,K) nonnegative

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
                   eta*sum(norms(Pp(:,:,k)+Pm(:,:,k) - Pp(:,:,j)-Pm(:,:,j),1));
            end
        end

        minimize(f0)
        subject to
            P == Pp-Pm;
            for k=1:K
                diag(Ao(:,:,k)) == 0;
                sum(Ao(:,1,k))== 1;
            end
    cvx_end
    norm_A = norm(vec(Ao),2)^2;
    diff_A = norm(vec(Ao-Ao_prev),2)^2/norm_A;
    Ao_prev = Ao;
%     figure()
%     imagesc(Ao(:,:,1))
    % Stop condition
    if diff_A < 1e-3
        if verb
            disp('Convergence achieved!')
        end
        break
    end
end