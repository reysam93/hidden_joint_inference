%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));


%%%% SETTING %%%%%
REW_ONLY_OBS = true;
C_TYPE = 'poly';  % or mrf

n_graphs = 25;
K = 3;
N = 20;
O = 19;
p = 0.2;
rew_links = 3;
F = 3;
M = 1e4;
sampled = true;
hid_nodes = 'min';
max_iters = 10;
th = 0.3;
verb_freq = 10;

deltas = [1e-2 1e-3 1e-4];
gammas = [10 25 50 75 100];
betas = [25 50 100];
etas = [25 50 100];
mus = [500 1000 2000];

err = zeros(length(mus),length(etas),length(betas),length(gammas),...
    length(deltas),n_graphs);
tic
parfor g=1:n_graphs
    % Generate data
    A = generate_connected_ER(N,p);
    [n_o, n_h] = select_hidden_nodes(hid_nodes,O,A);
    if REW_ONLY_OBS
        As = gen_similar_graphs_hid(A,Ks(end),pert_links,n_o,n_h);
    else
        As = gen_similar_graphs(A,K,rew_links);
    end

    Cs = create_cov(As,F,M,sampled, C_TYPE);
    Ao = As(n_o,n_o,:);
    Co = Cs(n_o,n_o,:);
    
    regs = struct();
    regs.alpha = 1;

    err_g = zeros(length(mus),length(etas),length(betas),...
        length(gammas),length(deltas));
    fsc_g = zeros(length(mus),length(etas),length(betas),...
        length(gammas),length(deltas));
    for o=1:length(deltas)
        regs.delta1 = deltas(o);
        for j=1:length(gammas)
            regs.gamma = gammas(j);
            for k=1:length(betas)
                regs.beta = betas(k);
                for f=1:length(etas)
                    regs.eta = etas(f);
                    for m=1:length(mus)
                        regs.mu = mus(m);


                        
                        [Ao_hat,~] = estA_pgl_colsp_rw(Co,regs,max_iters);
                        Ao_hat = Ao_hat./max(max(Ao_hat));
                        diff_Ao = Ao-Ao_hat;
                        for n=1:K
                            norm_Ao = norm(Ao(:,:,n),'fro');
                            err_g(m,f,k,j,o) = err_g(m,f,k,j,o) + ...
                                (norm(diff_Ao(:,:,n),'fro')/norm_Ao)^2/K;
                        end

                        if mod(g,verb_freq) == 1
                            disp(['Graph: ' num2str(g) ' delta: ' num2str(regs.delta1) ...
                                ' gamma: ' num2str(regs.gamma) ' beta: ' ...
                                num2str(regs.beta) ' eta: ' num2str(regs.eta) ...
                                ' mu: ' num2str(regs.mu) ' err: ' num2str(err_g(m,f,k,j,o))])
                        end

                    end
                end
            end
        end
    end
    err(:,:,:,:,:,g) = err_g;
end
t = toc;
disp(['--- ' num2str(t/3600) ' hours'])
%% Display results
mean_err = mean(err,6);
median_err = median(err,6);
[min_err, lin_idx] = min(mean_err(:));
[m,f,k,j,o] = ind2sub(size(mean_err),lin_idx);

disp(['Delta: ' num2str(deltas(o)) ' Gamma: ' num2str(gammas(j)) ' Beta: ' ...
    num2str(betas(k)) ' Eta: ' num2str(etas(f)) ' Mu: ' num2str(mus(m)) ...
    ' Err: ' num2str(mean_err(m,f,k,j,o))])

disp(['Min mean err: ' num2str(min(mean_err(:)))])
disp(['Min median err: ' num2str(min(median_err(:)))])