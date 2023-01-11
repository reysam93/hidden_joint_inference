rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

clear

% Exp parameters
nG = 50;
Prop_pert_links = [0, .05, .1, .15, .2, .25, .3];
K = 3;
N = 20;
O = 19;
H = N-O;
p = 0.2;
L = 3;
M = 1e4;
sampled = true;
hid_nodes = 'min';
max_iters = 10;
verb_freq = 10;

% REGS
regs_joint = struct();
regs_joint.alpha   = 1;      % Sparsity of S
regs_joint.gamma   = 75;    % Group Lasso (each P)
regs_joint.beta    = 10;      % Similarity of Ss
regs_joint.eta     = 50;      % Similarity of Ps
regs_joint.mu      = 1e4;    % Commutative penalty
regs_joint.delta1  = 1e-3;   % Small number for reweighted

regs_sep= struct();
regs_sep.alpha   = 1;      % Sparsity of S
regs_sep.gamma   = 75;    % Group Lasso
regs_sep.beta    = 0;      % Similarity of S
regs_sep.eta     = 0;      % Similarity of P
regs_sep.mu      = 1e4;    % Commutative penalty
regs_sep.delta1  = 1e-3;   % Small number for reweighted

models = {'Sep','Joint'};

%%
err = zeros(length(models),length(Prop_pert_links),nG);
err_no_sa = zeros(length(models),length(Prop_pert_links),nG);
tic
parfor g = 1:nG
    A = generate_connected_ER(N,p);
    [n_o, n_h] = select_hidden_nodes(hid_nodes,O,A);

    err_g = zeros(length(models),length(Prop_pert_links));
    err_no_sa_g = zeros(length(models),length(Prop_pert_links));
    for i = 1:length(Prop_pert_links)
        pert_links = round(sum(sum(A))/2*Prop_pert_links(i));

        if mod(g,verb_freq) == 1
            disp(['Graph: ' num2str(g) ' Rew links: ' num2str(pert_links)])
        end

        As = gen_similar_graphs(A,K,pert_links);
        Cs = create_cov(As,L,M,sampled);
        Ao = As(n_o,n_o,:);
        Co = Cs(n_o,n_o,:);

        % Sep-Hidden estimate
        [Ao_sep,~] = estA_pgl_colsp_rw(Co,regs_sep,max_iters);
        Ao_sep = Ao_sep./max(max(Ao_sep));

        % Joint-Hidden estimate
        [Ao_pgl,~] = estA_pgl_colsp_rw(Co,regs_joint,max_iters);
        Ao_pgl = Ao_pgl./max(max(Ao_pgl));

        % Compute error
        for k = 1:K
            Aok = Ao(:,:,k);
            norm_Aok = norm(Aok,'fro');
            Aok_norm = Aok/norm_Aok;

            err_g(1,i) = err_g(1,i) + (norm(Aok-Ao_sep(:,:,k),'fro')/norm_Aok)^2/K;
            err_g(2,i) = err_g(2,i) + (norm(Aok-Ao_pgl(:,:,k),'fro')/norm_Aok)^2/K;

            Ao_sep_norm = Ao_sep(:,:,k)/norm(Ao_sep(:,:,k),'fro');
            Ao_pgl_norm = Ao_pgl(:,:,k)/norm(Ao_pgl(:,:,k),'fro');
            err_no_sa_g(1,i) = err_no_sa_g(1,i) + norm(Aok_norm-Ao_sep_norm,'fro')^2/K;
            err_no_sa_g(2,i) = err_no_sa_g(2,i) + norm(Aok_norm-Ao_pgl_norm,'fro')^2/K;
        end

        if mod(g,verb_freq) == 1
            disp(['Graph: ' num2str(g) ' Rew links: ' num2str(pert_links) ' Err: ' num2str(err_g(2,i))])
        end

    end
    err(:,:,g) = err_g;
    err_no_sa(:,:,g) = err_no_sa_g;
end
t = toc;
disp(['----- ' num2str(t/60) ' mins -----'])

%%
mean_err = mean(err,3);
mean_err_no_sa = mean(err_no_sa,3);

figure()
plot(Prop_pert_links,mean_err)
legend(models)
xlabel('Prop. pert. links')

figure()
plot(Prop_pert_links,mean_err_no_sa)
legend(models)
xlabel('Prop. pert. links')

%%
median_err = median(err,3);
median_err_no_sa = median(err_no_sa,3);

figure()
plot(Prop_pert_links,median_err)
legend(models)
xlabel('Prop. pert. links')

figure()
plot(Prop_pert_links,median_err_no_sa)
legend(models)
xlabel('Prop. pert. links')









