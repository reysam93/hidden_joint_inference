function regs = get_regs(model,prms,H)
    delta1 = prms.delta1;
    delta2 = prms.delta2;
    epsilon = prms.epsilon;
    max_iters = prms.max_iters;
    if ~isfield(prms,'op')
        prms.op = 2;
    end
    
    regs = struct('delta1',delta1,'delta2',delta2,'epsilon',epsilon,'max_iters',max_iters);
    if strcmp(model,'lowrank')
        regs.alpha = 1e-2;
        regs.gamma = 1; 
        regs.beta = 1e-1;  
        regs.eta = 1; 
    elseif strcmp(model,'lowrank rw')
        regs.alpha = 1e-1;
        regs.gamma = 1e2; 
        regs.beta = 1;  
        regs.eta = 1;
        regs.mu = 1e2;    
    elseif strcmp(model,'PNN rw')
        if prms.op == 1
            regs.alpha = 1e-2;
            regs.beta = 1;
            regs.gamma = 100;
            regs.eta = 0.1;
            regs.mu = 1e2;
            regs.delta1 = 1e-3;
            regs.max_iters = 10;
        elseif prms.op == 2
        %best regs cov(randn coeff)
            regs.alpha = 1e-1;
            regs.beta = 1;
            regs.gamma = 100;
            regs.eta = 1e3;
            regs.mu = 1e2;
        end
    elseif strcmp(model,'grouplasso')
        regs.alpha = 1e-2;
        regs.gamma = 1;
        regs.mu = 1;
        regs.beta = 1;  
        regs.eta = 1;
    elseif strcmp(model,'PGL rw')
%         regs.alpha = 1e-2;%1e-2;
%         regs.beta = 1e-1;%1e-1; 
%         regs.eta = 1e-3;%1e-2;
%         regs.mu = 1e2;%1e1;
%         if H == 1
%             regs.gamma = 5;
%         elseif H == 2
%             regs.gamma = 2.64;
%         else
%             regs.gamma = 3.79;
%         end
        regs.alpha = 1;
        regs.beta = 10;
        regs.gamma = 100;
        regs.eta = 10;
        regs.mu = 1e4;
        
    elseif strcmp(model,'baseline')
        regs.alpha = 1e-2; 
        regs.beta = 1e2;
        regs.epsilon = 1e-1;
    elseif strcmp(model,'No hidden rw')
        regs.alpha = 1e-2; 
        regs.beta = 1e-1;
        regs.mu = 1;
    else 
        disp('Unknown algorithm')
    end
end


