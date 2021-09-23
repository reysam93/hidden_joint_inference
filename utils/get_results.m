function [err_joint,err_sep] = get_results(A,mdl)
    K = size(A,4);
    err_joint = zeros(K,1);
    err_sep = zeros(K,1);
    
    Ao = squeeze(A(1,:,:,:));
    Ao_hat = squeeze(A(2,:,:,:));
    Ao_hat_sep = squeeze(A(3,:,:,:));
    
    Ao_hat = Ao_hat./max(max(Ao_hat));
    Ao_hat_sep = Ao_hat_sep./max(max(Ao_hat_sep));
    for k=1:K
        if strcmp(mdl,'fronorm')
            norm_Ao = norm(Ao(:,:,k),'fro')^2;
            err_joint(k) = norm(Ao(:,:,k)-Ao_hat(:,:,k),'fro')^2/norm_Ao;
            err_sep(k) = norm(Ao(:,:,k)-Ao_hat_sep(:,:,k),'fro')^2/norm_Ao;
        elseif strcmp(mdl,'fscore')
            [~,~,err_joint(k),~,~] = graph_learning_perf_eval(Ao(:,:,k),Ao_hat(:,:,k));
            [~,~,err_sep(k),~,~] = graph_learning_perf_eval(Ao(:,:,k),Ao_hat_sep(:,:,k));
        end
    end

end