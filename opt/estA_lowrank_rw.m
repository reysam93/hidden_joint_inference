function [Ao,P] = estA_lowrank_rw(Co, regs)
Ao = [];
P = [];

O = size(Co,1);
K = size(Co,3);

alpha = regs.alpha;%1e-2
gamma = regs.gamma;%1e1
beta = regs.beta;%1e-1
eta = regs.eta;%1e-1
mu = regs.mu;%100
max_iters = regs.max_iters;
del1 = regs.delta1;

Ao_prev = ones(O,O,K);

for i=1:max_iters
    W_Ao = alpha*ones(O,O,K)./(Ao_prev+del1);
    
    cvx_begin quiet
        variable Ao(O,O,K) symmetric nonnegative
        variable P(O,O,K)

        f0 = vec(W_Ao)'*vec(Ao);
        for k=1:K
            f0 = f0 + gamma*norm_nuc(P(:,:,k)) + mu*norm(Co(:,:,k)*Ao(:,:,k)+P(:,:,k)-Ao(:,:,k)*Co(:,:,k)-P(:,:,k)','fro');
            for j=1:(k-1)
               f0 = f0 + beta*norm(vec(Ao(:,:,k)-Ao(:,:,j)),1) + eta*sum(norms(P(:,:,k)-P(:,:,j),2));
            end
        end

        minimize(f0)
        subject to
            %norm(vec(pagemtimes(Co,Ao)+P-pagemtimes(Ao,Co)-P'),2)^2 <= eps^2;
            for k=1:K
                diag(Ao(:,:,k)) == 0;
                sum(Ao(:,1,k))== 1;
            end
    cvx_end
    %disp([num2str(norm(Ao_prev(:,:,1)-Ao(:,:,1),'fro')),'---', num2str(norm(Co(:,:,1)*Ao(:,:,1)+P(:,:,1)-Ao(:,:,1)*Co(:,:,1)-P(:,:,1)','fro'))])
    Ao_prev = Ao;
end