function [S, out] = LVGLASSO(C, reg, verb)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %params para covarianza python alpha= 1e-3; beta=1e-3;
    
    %params para covarianza poly alpha = 1; beta = 1e-3;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    alpha = reg.alpha;
    beta = reg.beta;
    rho = 1/size(C,1);
    opts.continuation = 1; opts.num_continuation = 0;
    opts.eta = 1/rho; opts.muf = 1e-6;
    opts.maxiter = 500; opts.stoptol = 1e-5;
    opts.over_relax_par = 1;

    n = size(C,1);
    opts.mu = n;

    tic; out = ADMM_B(C,alpha,beta,opts); out.elapsed_time = toc;
    if verb
        fprintf('ADMM_B: obj: %e, iter: %d \n',out.obj,out.iter);
    end
    out.res = out.resid;
    S = out.S-diag(diag(out.S));
end
