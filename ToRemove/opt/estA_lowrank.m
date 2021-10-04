function [Ao,P] = estA_lowrank(Co, regs)
Ao = [];
P = [];

O = size(Co,1);
K = size(Co,3);
alp = regs.alpha; %1e-3
gamma = regs.gamma;%1e1
beta = regs.beta;%1
eta = regs.eta;%1e-1
eps = regs.epsilon;%1e-6
cvx_begin quiet
    variable Ao(O,O,K) symmetric nonnegative
    variable P(O,O,K)

    f0 = 0;
    for k=1:K
        f0 = f0 + alp*norm(vec(Ao(:,:,k)),1) + gamma*norm_nuc(P(:,:,k));
        for j=1:(k-1)
           f0 = f0 + beta*norm(vec(Ao(:,:,k)-Ao(:,:,j)),1) + eta*sum(norms(P(:,:,k)-P(:,:,j),2));
        end
    end
    
    minimize(f0)
    subject to
        for k=1:K
            diag(Ao(:,:,k)) == 0;
            sum(Ao(:,1,k))== 1;
            norm(Co(:,:,k)*Ao(:,:,k)+P(:,:,k)-Ao(:,:,k)*Co(:,:,k)-P(:,:,k)','fro') <= eps;
        end
cvx_end