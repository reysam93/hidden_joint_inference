function [Ao,Aoh,Coh,A_init] = estA_alt(Co,H,regs,verb)
if nargin < 4
    verb = false;
end

O = size(Co,1);
K = size(Co,3);
alpha = regs.alpha;
beta = regs.beta;
eta = regs.eta;
lamb = regs.lambda;
mu = regs.mu;
max_iters = 5;

Aoh = zeros(O,H,K);
Coh = zeros(O,H,K);

if verb
    disp('Initializing')
end

% Missing initialization
[Ao,P] = estA_lowrank(Co,regs);
A_init = Ao;
for k=1:K
    [U,Sigma,V]=svd(P(:,:,k));
    Aoh(:,:,k) = U(:,1:H)*Sigma(1:H,1:H);
    Coh(:,:,k) = V(:,1:H)';
end
Aoh(Aoh<0) = 0;

disp('Starting iterative algorithm')
for i=1:max_iters
    % Step 1: infer Ao and Aoh
    cvx_begin quiet
        variable Ao(O,O,K) symmetric
        variable Aoh(O,H,K)

        f0 = 0;
        for k=1:K
            % Sparse penalties
            f0 = f0 + alpha*norm(vec(Ao(:,:,k)),1) + lamb*norm(vec(Aoh(:,:,k)),1);
            % Commutativity penalty
            f0 = f0 + mu*norm(Co(:,:,k)*Ao(:,:,k)+Coh(:,:,k)*Aoh(:,:,k)'...
                -Ao(:,:,k)*Co(:,:,k)-Aoh(:,:,k)*Coh(:,:,k)','fro');
            % Graph similarity penalties
            for j=2:k
               f0 = f0 + beta*norm(vec(Ao(:,:,k)-Ao(:,:,j)),1)...
                   + eta*norm(vec(Aoh(:,:,k)-Aoh(:,:,j)));
            end
        end

        % Vectorize objective function
        %f0 = alpha*norm(vec(Ao),1) + lamb*norm(vec(Aoh),1);
        %norm(vec(pagemtimes(Co,Ao)),2)

        minimize(f0)
        subject to
            Ao >= 0;
            Aoh >= 0;
            sum(Ao(:,1,1))== 1;
            for k=1:K
                diag(Ao(:,:,k)) == 0;
            end
    cvx_end

    % Step 2: infer Coh
    cvx_begin quiet
        variable Coh(O,H,K)

        f0 = 0;
        for k=1:K
            % Commutativity penalty
            f0 = f0 + mu*norm(Co(:,:,k)*Ao(:,:,k)+Coh(:,:,k)*Aoh(:,:,k)'...
                -Ao(:,:,k)*Co(:,:,k)-Aoh(:,:,k)*Coh(:,:,k)','fro');
        end
        minimize(f0)        
    cvx_end
    
    if verb
        disp(['Iter: ' num2str(i)])
    end
end