%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

n_graphs = 16;
K = 3;
N = 20;
O = 19;
p = 0.2;
pert_links = 3;
L = 3;
M = 1e3;
sampled = false;
hid_nodes = 'min';
max_iters = 15;

alphas = [0.1 1 10];
lambdas = [0.1 1 10];
betas = [10 50 100];
etas = [10 50 100];
mus = [10 100 1000];

delta1 = 1e-4;
delta2 = 1e-4;

err = zeros(length(mus),length(etas),length(betas),length(lambdas),...
    length(alphas),n_graphs);
tic
parfor g=1:n_graphs
    disp(['Graph: ' num2str(g)])
    A = generate_connected_ER(N,p);
    As = gen_similar_graphs(A,K,pert_links);
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
    
    % Create covariances
    Cs = zeros(N,N,K);
    for k=1:K
        h = rand(L,1)*2-1;
        H = zeros(N);
        for l=1:L
            H = H + h(l)*As(:,:,k)^(l-1);
        end
        
        if sampled
            X = H*randn(N,M);
            Cs(:,:,k) = X*X'/M;
        else
            Cs(:,:,k) = H^2;
        end
    end
    
    Ao = As(n_o,n_o,:);
    Aoh = As(n_o,n_h,:);
    Co = Cs(n_o,n_o,:);
    Coh = Cs(n_o,n_h,:);
    
    regs = struct('delta1',delta1,'delta2',delta2);
    err_g = zeros(K,length(mus),length(etas),length(betas),...
        length(lambdas),length(alphas));
    for i=1:length(alphas)
        regs.alpha = alphas(i);
        for j=1:length(lambdas)
            regs.lambda = lambdas(j);
            for k=1:length(betas)
                regs.beta = betas(k);
                for l=1:length(etas)
                    regs.eta = etas(l);
                    for m=1:length(mus)
                        regs.mu = mus(m);
                        [Ao_hat,~,~] = estA_alt_rw(Co,N-O,regs,max_iters);
                        Ao_hat = Ao_hat./max(max(Ao_hat));
                        diff_Ao = Ao-Ao_hat;
                        for n=1:K
                            norm_Ao = norm(Ao(:,:,n),'fro')^2;
                            err_g(n,m,l,k,j,i) = norm(diff_Ao(:,:,n),'fro')^2/norm_Ao;
                        end
                    end
                end
            end
        end
    end
    err(:,:,:,:,:,g) = mean(err_g,1);
end
t = toc;
disp(['--- ' num2str(t/3600) ' hours'])
%%
mean_err = mean(err,6)
median_err = median(err,6)