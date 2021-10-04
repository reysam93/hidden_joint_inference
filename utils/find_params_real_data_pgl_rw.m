%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

signal_trials = 30;
K = 3;
N = 32;
O = 31;
F = 3;
M = 1e4;
hid_nodes = 'min';
max_iters = 10;
th = 0.3;
graphs = [7 9 12];

deltas = [1e-2 1e-3 1e-4];
gammas = [50 100 250];
betas = [10 50 100];
etas = [10 50 100];
mus = [250 500 1000];

As = get_student_networks_graphs(graphs,N);

[n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
Ao = As(n_o,n_o,:);
Aoh = As(n_o,n_h,:);

err = zeros(length(mus),length(etas),length(betas),length(gammas),...
    length(deltas),signal_trials);
fsc = zeros(length(mus),length(etas),length(betas),length(gammas),...
    length(deltas),signal_trials);
tic
parfor g=1:signal_trials
    disp(['Trial: ' num2str(g)])
    A = generate_connected_ER(N,p);
    As = gen_similar_graphs(A,K,pert_links);
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
    
    % Create covariances
    Cs = zeros(N,N,K);
    for k=1:K
        h = rand(F,1)*2-1;
        H = zeros(N);
        for f=1:F
            H = H + h(f)*As(:,:,k)^(f-1);
        end
        
        X = H*randn(N,M);
        Cs(:,:,k) = X*X'/M;

    end
    Co = Cs(n_o,n_o,:);
    
    regs = struct();
    regs.alpha = 1;
    err_g = zeros(K,length(mus),length(etas),length(betas),...
        length(gammas),length(deltas));
    fsc_g = zeros(K,length(mus),length(etas),length(betas),...
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
                        [Ao_hat,~] = estA_pgl_colsp_rw2(Co,N-O,regs,max_iters);
                        Ao_hat = Ao_hat./max(max(Ao_hat));
                        diff_Ao = Ao-Ao_hat;
                        for n=1:K
                            norm_Ao = norm(Ao(:,:,n),'fro')^2;
                            err_g(n,m,f,k,j,o) = norm(diff_Ao(:,:,n),'fro')^2/norm_Ao;
                            
                            Ao_th = Ao(:,:,n);
                            Ao_th(Ao_th >= th) = 1;
                            Ao_th(Ao_th < th) = 0;
                            
                            Ao_hat_th = Ao_hat(:,:,n);
                            Ao_hat_th(Ao_hat_th >= th) = 1;
                            Ao_hat_th(Ao_hat_th < th) = 0;
                            [~,~,fsc_g(n,m,f,k,j,o),~,~] = ...
                                graph_learning_perf_eval(Ao_th,Ao_hat_th);
                        end
                    end
                end
            end
        end
    end
    err(:,:,:,:,:,g) = mean(err_g,1);
    fsc(:,:,:,:,:,g) = mean(fsc_g,1);
end
t = toc;
disp(['--- ' num2str(t/3600) ' hours'])
%% Display results
mean_err = mean(err,6);
median_err = median(err,6);
mean_fsc = mean(fsc,6);
median_fsc = median(fsc,6);
rec_graphs = sum(fsc==1,6)/signal_trials;

disp(['Min mean err: ' num2str(min(mean_err(:)))])
disp(['Min median err: ' num2str(min(median_err(:)))])
disp(['Max mean fsc: ' num2str(max(mean_fsc(:)))])
disp(['Max median err: ' num2str(max(median_fsc(:)))])
disp(['Max rec: ' num2str(max(rec_graphs(:)))])