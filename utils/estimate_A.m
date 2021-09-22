function [Ao_hat, Ao_hat_sep] = estimate_A(model,Co,regs)
    Ao_hat = zeros(size(Co));
    Ao_hat_sep = zeros(size(Co));
    K = size(Co,3);
    if strcmp(model,'lowrank')
        [Ao_hat,~] = estA_lowrank(Co,regs);
        for k=1:K
            [Ao_hat_sep(:,:,k),~] = estA_lowrank(Co(:,:,k),regs);
        end
    elseif strcmp(model,'lowrank rw')
        [Ao_hat,~] = estA_lowrank_rw(Co,regs);
        for k=1:K
            [Ao_hat_sep(:,:,k),~] = estA_lowrank_rw(Co(:,:,k),regs);
        end
    elseif strcmp(model,'baseline')
        [Ao_hat,~] = estA_baseline(Co,regs);
        for k=1:K
            [Ao_hat_sep(:,:,k),~] = estA_baseline(Co(:,:,k),regs);
        end
    elseif strcmp(model,'grouplasso')
        [Ao_hat,~] = estA_grouplasso(Co,regs);
        for k=1:K
            [Ao_hat_sep(:,:,k),~] = estA_grouplasso(Co(:,:,k),regs);
        end
    elseif strcmp(model,'grouplasso rw')
        [Ao_hat,~] = estA_grouplasso_rw(Co,regs);
        for k=1:K
            [Ao_hat_sep(:,:,k),~] = estA_grouplasso_rw(Co(:,:,k),regs);
        end
    end
    



end