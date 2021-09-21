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
max_iters = 10;

% Aoh = zeros(O,H,K);
% Coh = zeros(O,H,K);

if verb
    disp('Initializing')
end

% Missing initialization
[Ao,P] = estA_lowrank(Co,regs);
A_init = Ao;
for k=1:K
    [~,~,V] = svd(P(:,:,k));
%     [U,Sigma,V] = svd(P(:,:,k));
%     Aoh(:,:,k) = U(:,1:H)*Sigma(1:H,1:H);
    Coh(:,:,k) = V(:,1:H)';
end
% Aoh(Aoh<0) = 0;
Coh = randn(O,H,K);

disp('Starting iterative algorithm')
Ao_prev = A_init;
for i=1:max_iters
    % Step 1: infer Ao and Aoh
    cvx_begin quiet
        variable Ao(O,O,K) symmetric nonnegative
        variable Aoh(O,H,K) nonnegative

        f1 = 0;
        for k=1:K
            % Sparse penalties
            f1 = f1 + alpha*norm(vec(Ao(:,:,k)),1) + lamb*norm(vec(Aoh(:,:,k)),1);
            % Commutativity penalty
            f1 = f1 + mu*norm(Co(:,:,k)*Ao(:,:,k)+Coh(:,:,k)*Aoh(:,:,k)'...
                -Ao(:,:,k)*Co(:,:,k)-Aoh(:,:,k)*Coh(:,:,k)','fro');
            % Graph similarity penalties
            for j=1:(k-1)
               f1 = f1 + beta*norm(vec(Ao(:,:,k)-Ao(:,:,j)),1)...
                   + eta*norm(vec(Aoh(:,:,k)-Aoh(:,:,j)),1);
            end
        end

        minimize(f1)
        subject to
            for k=1:K
                diag(Ao(:,:,k)) == 0;
                sum(Ao(:,1,k))== 1;
            end
    cvx_end

    % Step 2: infer Coh
    cvx_begin quiet
        variable Coh(O,H,K)

        f2 = 0;
        for k=1:K
            % Commutativity penalty
            f2 = f2 + mu*norm(Co(:,:,k)*Ao(:,:,k)+Coh(:,:,k)*Aoh(:,:,k)'...
                -Ao(:,:,k)*Co(:,:,k)-Aoh(:,:,k)*Coh(:,:,k)','fro');
        end
        minimize(f2)        
    cvx_end
    
    norm_Ao_prev = norm(vec(Ao_prev),2)^2;
    diff_Ao = norm(vec(Ao-Ao_prev),2)^2/norm_Ao_prev;
    comm = norm(Co(:,:,k)*Ao(:,:,k)+Coh(:,:,k)*Aoh(:,:,k)'...
                -Ao(:,:,k)*Co(:,:,k)-Aoh(:,:,k)*Coh(:,:,k)','fro');
    
    if verb
        disp(['Iter: ' num2str(i) '   Ao-Ao_prev: ' num2str(diff_Ao)...
            '   Comm: ' num2str(comm)])
    end
end