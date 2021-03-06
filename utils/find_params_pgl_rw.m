%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

n_graphs = 30;
K = 3;
N = 20;
O = 19;
p = 0.2;
pert_links = 3;
F = 3;
M = 1e4;
sampled = true;
hid_nodes = 'min';
max_iters = 10;
th = 0.3;
Cmrf = true;

deltas = [1e-2 1e-3 1e-4];
gammas = [10 25 50 75 100];
betas = [25 50 100];
etas = [25 50 100];
mus = [500 1000 2000];

err = zeros(length(mus),length(etas),length(betas),length(gammas),...
    length(deltas),n_graphs);
fsc = zeros(length(mus),length(etas),length(betas),length(gammas),...
    length(deltas),n_graphs);
tic
parfor g=1:n_graphs
    disp(['Graph: ' num2str(g)])
    A = generate_connected_ER(N,p);
    As = gen_similar_graphs(A,K,pert_links);
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
    
    % Create covariances
    Cs = zeros(N,N,K);
    for k=1:K
        if Cmrf
            eigvals = eig(As(:,:,k));
            C_inv = (0.01-min(eigvals))*eye(N,N) + (0.9+0.1*rand(1,1))*As(:,:,k);
            C_true = inv(C_inv);
        else
            h = rand(F,1)*2-1;
            H = zeros(N);
            for f=1:F
                H = H + h(f)*As(:,:,k)^(f-1);
            end
            C_true = H^2;
        end
        
        if sampled
            X = sqrtm(C_true)*randn(N,M);
            Cs(:,:,k) = X*X'/M;
        else
            Cs(:,:,k) = C_true;
        end
    end
    
    Ao = As(n_o,n_o,:);
    Aoh = As(n_o,n_h,:);
    Co = Cs(n_o,n_o,:);
    Coh = Cs(n_o,n_h,:);
    
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
                        [Ao_hat,~] = estA_pgl_colsp_rw(Co,regs,max_iters);
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
rec_graphs = sum(fsc==1,6)/n_graphs;

disp(['Min mean err: ' num2str(min(mean_err(:)))])
disp(['Min median err: ' num2str(min(median_err(:)))])
disp(['Max mean fsc: ' num2str(max(mean_fsc(:)))])
disp(['Max median err: ' num2str(max(median_fsc(:)))])
disp(['Max rec: ' num2str(max(rec_graphs(:)))])