function [Ao,P] = estA_grouplasso(Co, regs)
Ao = [];
P = [];

O = size(Co,1);
K = size(Co,3);
alp = regs.alpha; %1e-2
gamma = regs.gamma;%21.54
beta = regs.beta;%2.154
eta = regs.eta;%1
eps = regs.epsilon;%1e-6
cvx_begin quiet
    variable Ao(O,O,K) symmetric nonnegative
    variable P(O,O,K)

    f0 = 0;
    for k=1:K
        f0 = f0 + alp*norm(vec(Ao(:,:,k)),1) + gamma*sum(norms(P(:,:,k),2));
        for j=1:(k-1)
           f0 = f0 + beta*norm(vec(Ao(:,:,k)-Ao(:,:,j)),1) + eta*norm(vec(P(:,:,k)-P(:,:,j)),1);
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