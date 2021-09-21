function [Ao,P,A_init] = estA_pnn_colsp(Co,H,regs,verb)
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

% disp('Starting iterative algorithm')
Ao_prev = ones(O,O,K);
Aoh_prev = ones(O,H,K);
for i=1:max_iters
    % Step 1: infer Ao and Aoh
    cvx_begin quiet
        variable Ao(O,O,K) symmetric nonnegative
        variable P(O,O,K)
        variable Pm(O,O,K) nonnegative
        variable Pp(O,O,K) nonnegative

        f0 = 0;
        for k=1:K
            % Sparse penalties
            f0 = f0 + alpha*norm(vec(Ao(:,:,k)),1) + gamma*norm_nuc(P(:,:,k));
            % Commutativity penalty
            f0 = f0 + mu*norm(Co(:,:,k)*Ao(:,:,k)+P(:,:,k)...
                -Ao(:,:,k)*Co(:,:,k)-P(:,:,k)','fro');
            % Graph similarity penalties
            for j=1:(K-1)
               f0 = f0 + beta*norm(vec(Ao(:,:,k)-Ao(:,:,j)),1);
               for c=1:O
                   f0 = f0 + eta*norm( Pp(:,c,k)+Pm(:,c,k) - Pp(:,c,j)-Pm(:,c,j) ,1);
               end
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
    
    norm_Ao_prev = norm(vec(Ao_prev),2)^2;
    diff_Ao = norm(vec(Ao-Ao_prev),2)^2/norm_Ao_prev;
    Ao_prev = Ao;
    
    comm = norm(Co(:,:,k)*Ao(:,:,k)+P(:,:,k)...
                -Ao(:,:,k)*Co(:,:,k)-P(:,:,k)','fro');
    
    if verb
        disp(['Iter: ' num2str(i) '   Ao-Ao_prev: ' num2str(diff_Ao)...
            '   Comm: ' num2str(comm)])
    end
end