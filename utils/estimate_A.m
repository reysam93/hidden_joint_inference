function [Ao_hat, Ao_hat_sep] = estimate_A(model,Co,regs)
    Ao_hat = zeros(size(Co));
    Ao_hat_sep = zeros(size(Co));
    K = size(Co,3);

    if strcmp(model,'No hidden rw')
        [Ao_hat] = estA_no_hidden_rw(Co,regs);
%         for k=1:K
%             [Ao_hat_sep(:,:,k),~] = estA_baseline_rw(Co(:,:,k),regs);
%         end
    elseif strcmp(model,'PNN rw')
        [Ao_hat,~] = PNN_rw(Co,regs);
%         for k=1:K
%             [Ao_hat_sep(:,:,k),~] = estA_lowrank_rw(Co(:,:,k),regs);
%         end
    elseif strcmp(model,'PGL rw')
        [Ao_hat,~] = estA_pgl_colsp_rw(Co,regs);
%         for k=1:K
%             [Ao_hat_sep(:,:,k),~] = estA_grouplasso_rw(Co(:,:,k),regs);
%         end
    elseif strcmp(model,'lowrank rw')
        [Ao_hat,~] = estA_lowrank_rw(Co,regs);
%         for k=1:K
%             [Ao_hat_sep(:,:,k),~] = estA_lowrank_rw(Co(:,:,k),regs);
%         end
    elseif strcmp(model,'baseline')
        [Ao_hat,~] = estA_baseline(Co,regs);
        for k=1:K
            [Ao_hat_sep(:,:,k),~] = estA_baseline(Co(:,:,k),regs);
        end
    else
        disp('Unknown algorithm')
    end
    



end