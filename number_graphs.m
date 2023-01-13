%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

clear

% Exp parameters
nG = 100;     
Ks = 1:6;
N = 20;
O = 19;
H = N-O;
p = 0.2;
pert_links = 3;
L = 3;
M = 1e4;
sampled = true;
hid_nodes = 'min';
max_iters = 10;
verb_freq = 20;

% Weight for regularization
regs_joint = struct(); 
regs_joint.alpha   = 1;      % Sparsity of S
regs_joint.gamma   = 100;    % Group Lasso (each P)
regs_joint.beta    = 10;      % Similarity of Ss
regs_joint.eta     = 10;      % Similarity of Ps
regs_joint.mu      = 1e4;    % Commutative penalty
regs_joint.delta1  = 1e-3;   % Small number for reweighted

regs_sep= struct();
regs_sep.alpha   = 1;      % Sparsity of S
regs_sep.gamma   = 100;    % Group Lasso
regs_sep.beta    = 0;      % Similarity of S
regs_sep.eta     = 0;      % Similarity of P
regs_sep.mu      = 1e4;    % Commutative penalty
regs_sep.delta1  = 1e-3;   % Small number for reweighted

models = {'Sep, Hid','Joint, Hid','Sep, No Hid', 'Joint, No Hid'};
% models = {'Sep, Hid','Joint, Hid'};


%%

%As = zeros(N,N,Ks(end));
err = zeros(length(models),length(Ks),nG);
err2 = zeros(length(models),length(Ks),nG);
tic
parfor g = 1:nG
    % Create graphs
    A = generate_connected_ER(N,p);
    As = gen_similar_graphs(A,Ks(end),pert_links);
    
    % Create covariance
    Cs = create_cov(As,L,M,sampled);

    % Select hidden
    [n_o, n_h] = select_hidden_nodes(hid_nodes,O,As(:,:,1));

    err_g = zeros(length(models),length(Ks));
    err2_g = zeros(length(models),length(Ks));
    for i = 1:length(Ks)
        K = Ks(i);
        Ao = As(n_o,n_o,1:K);
        Co = Cs(n_o,n_o,1:K);

        % Sep-Hidden estimate
        [Ao_sep,~] = estA_pgl_colsp_rw(Co,regs_sep,max_iters);
        Ao_sep_max1 = Ao_sep./max(max(Ao_sep));

        % Joint-Hidden estimate
        [Ao_pgl,~] = estA_pgl_colsp_rw(Co,regs_joint,max_iters);
        Ao_pgl_max1 = Ao_pgl./max(max(Ao_pgl));

        % Sep-No hidden
        Ao_sep_n = estA_stat_nohid(Co,regs_sep,max_iters);
        Ao_sep_n_max1 = Ao_sep_n./max(max(Ao_sep_n));

        % Joint No-Hidden
        Ao_pgl_n = estA_stat_nohid(Co,regs_joint,max_iters);
        Ao_pgl_n_max1 = Ao_pgl_n./max(max(Ao_pgl_n));

        % Compute error
        for k = 1:K
            Aok = Ao(:,:,k);
            norm_Aok = norm(Aok,'fro');
            Aok_norm = Aok/norm_Aok;

            err_g(1,i) = err_g(1,i) + (norm(Aok-Ao_sep_max1(:,:,k),'fro')/norm_Aok)^2/K;
            err_g(2,i) = err_g(2,i) + (norm(Aok-Ao_pgl_max1(:,:,k),'fro')/norm_Aok)^2/K;
            err_g(3,i) = err_g(3,i) + (norm(Aok-Ao_sep_n_max1(:,:,k),'fro')/norm_Aok)^2/K;
            err_g(4,i) = err_g(4,i) + (norm(Aok-Ao_pgl_n_max1(:,:,k),'fro')/norm_Aok)^2/K;

            Ao_sep_norm = Ao_sep(:,:,k)/norm(Ao_sep(:,:,k),'fro');
            Ao_pgl_norm = Ao_pgl(:,:,k)/norm(Ao_pgl(:,:,k),'fro');
            Ao_sep_n_norm = Ao_sep_n(:,:,k)/norm(Ao_sep_n(:,:,k),'fro');
            Ao_pgl_n_norm = Ao_pgl_n(:,:,k)/norm(Ao_pgl_n(:,:,k),'fro');
            err2_g(1,i) = err2_g(1,i) + norm(Aok_norm-Ao_sep_norm,'fro')^2/K;
            err2_g(2,i) = err2_g(2,i) + norm(Aok_norm-Ao_pgl_norm,'fro')^2/K;
            err2_g(3,i) = err2_g(3,i) + norm(Aok_norm-Ao_sep_n_norm,'fro')^2/K;
            err2_g(4,i) = err2_g(4,i) + norm(Aok_norm-Ao_pgl_n_norm,'fro')^2/K;
        end

        if mod(g,verb_freq) == 1
            disp(['Graph: ' num2str(g) ' K: ' num2str(K) ' Err: ' num2str(err_g(2,i))])
        end

    end
    err(:,:,g) = err_g;
    err2(:,:,g) = err2_g;
end
t = toc;
disp(['----- ' num2str(t/60) ' mins -----'])

%%
mean_err = mean(err,3);
mean_err2 = mean(err2,3);

figure()
plot(Ks,mean_err)
legend(models)
xlabel('K')
ylabel('Mean err')

figure()
plot(Ks,mean_err2)
legend(models)
xlabel('K')
ylabel('Mean err2')


%%
mean_err = median(err,3);
mean_err2 = median(err2,3);
mean_err_no_sa = median(err_no_sa,3);

figure()
plot(Ks,mean_err)
legend(models)
xlabel('K')
ylabel('Median err')

figure()
plot(Ks,mean_err2)
legend(models)
xlabel('K')
ylabel('Median err2')