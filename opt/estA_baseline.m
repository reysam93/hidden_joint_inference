function [Ao,P] = estA_baseline(Co, regs)
Ao = [];
P = [];

O = size(Co,1);
K = size(Co,3);
                        %no hidden      %hidden     
alp = regs.alpha;       %1              %1e-2
beta = regs.beta;       %1              %1e2
eps = regs.epsilon;     %1e-6           %1e-1
cvx_begin quiet
    variable Ao(O,O,K) symmetric nonnegative
    variable P(O,O,K)

    f0 = 0;
    for k=1:K
        f0 = f0 + alp*norm(vec(Ao(:,:,k)),1);
        for j=1:(k-1)
           f0 = f0 + beta*norm(vec(Ao(:,:,k)-Ao(:,:,j)),1);
        end
    end
    
    minimize(f0)
    subject to
        for k=1:K
            diag(Ao(:,:,k)) == 0;
            sum(Ao(:,1,k))== 1;
            norm(Co(:,:,k)*Ao(:,:,k)-Ao(:,:,k)*Co(:,:,k),'fro') <= eps;
        end
cvx_end
end