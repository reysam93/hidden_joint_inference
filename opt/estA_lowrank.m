function [Ao,P] = estA_lowrank(Co, regs)
Ao = [];
P = [];

O = size(Co,1);
K = size(Co,3);
alp = regs.alpha;
gamma = regs.gamma;
eps = regs.epsilon;

cvx_begin quiet
    variable Ao(O,O,K) symmetric
    variable P(O,O,K)

    f0 = 0;
    for k=1:K
        f0 = f0 + alp*norm(vec(Ao(:,:,k)),1) + gamma*norm_nuc(P(:,:,k));
        
        for j=2:k
           f0 = f0 + norm(vec(Ao(:,:,k)-Ao(:,:,j)),1);
        end
    end

    minimize(f0)
    subject to
        Ao >= 0;
        sum(Ao(:,1,1))== 1;
        for k=1:K
            diag(Ao(:,:,k)) == 0;
            norm(Co(:,:,k)*Ao(:,:,k)+P(:,:,k)-Ao(:,:,k)*Co(:,:,k)-P(:,:,k)','fro') <= eps;
        end
cvx_end