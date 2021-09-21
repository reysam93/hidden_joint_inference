function [Ao,Aoh,Coh] = estA_alt_rw(Co,H,regs,max_iters,verb)
if nargin < 4
    max_iters = 10;
end
if nargin < 5
    verb = false;
end

O = size(Co,1);
K = size(Co,3);

alpha = regs.alpha;
beta = regs.beta;
eta = regs.eta;
lamb = regs.lambda;
mu = regs.mu;
del1 = regs.delta1;
del2 = regs.delta2;

if verb
    disp('Starting iterative algorithm')
end

norm_Co = norm(vec(Co),2)^2;

Coh = randn(O,H,K);
Ao_prev = ones(O,O,K);
Aoh_prev = ones(O,H,K);
for i=1:max_iters
    % Step 1: infer Ao and Aoh    
    W_Ao = alpha*ones(O,O,K)./(Ao_prev+del1);
    W_Aoh = lamb*ones(O,H,K)./(Aoh_prev+del2);
    cvx_begin quiet
        variable Ao(O,O,K) symmetric nonnegative
        variable Aoh(O,H,K) nonnegative
        % Reweighted Sparse penalties
        f1 = vec(W_Ao)'*vec(Ao) + vec(W_Aoh)'*vec(Aoh);
        for k=1:K
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
%                 sum(Ao(:,1,k))>= 1;
%                 sum(Aoh(:,1,k))>= 1;
            end
    cvx_end
    
    norm_Ao = norm(vec(Ao),2)^2;
    diff_Ao = norm(vec(Ao-Ao_prev),2)^2/norm_Ao;
    Ao_prev = Ao;
    Aoh_prev = Aoh;
    
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

    shape = [2 1 3];
    comm = norm(vec(pagemtimes(Co,Ao)+pagemtimes(Coh,permute(Aoh,shape))...
        -pagemtimes(Ao,Co)-pagemtimes(Aoh,permute(Coh,shape))),2)^2;
    comm_norm = comm/norm_Ao/norm_Co;
    if verb
        disp(['Iter: ' num2str(i) '   Ao-Ao_prev: ' num2str(diff_Ao)...
            '   Comm: ' num2str(comm) '   Comm norm: ' num2str(comm_norm)])
    end
    
    % Stop condition
    if diff_Ao < 1e-3
        if verb
            disp('Convergence achieved!')
        end
        break
    end
end