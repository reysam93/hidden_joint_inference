function [Ao,P] = estA_lowrank(Co, regs)
Ao = [];
P = [];

O = size(Co,1);
K = size(Co,3);
alp = regs.alpha;
gamma = regs.gamma;
beta = regs.beta;
eps = regs.epsilon;

cvx_begin quiet
    variable Ao(O,O,K) symmetric nonnegative
    variable P(O,O,K)

    f0 = 0;
    for k=1:K
        f0 = f0 + alp*norm(vec(Ao(:,:,k)),1) + gamma*norm_nuc(P(:,:,k));
        for j=1:(k-1)
           f0 = f0 + norm(vec(Ao(:,:,k)-Ao(:,:,j)),1);
        end
    end
    
    minimize(f0)
    subject to
        %norm(vec(pagemtimes(Co,Ao)+P-pagemtimes(Ao,Co)-P'),2)^2 <= eps^2;
        for k=1:K
            diag(Ao(:,:,k)) == 0;
            sum(Ao(:,1,k))== 1;
            norm(Co(:,:,k)*Ao(:,:,k)+P(:,:,k)-Ao(:,:,k)*Co(:,:,k)-P(:,:,k)','fro') <= eps;
        end
cvx_end