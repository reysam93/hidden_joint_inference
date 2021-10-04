%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

n_graphs = 18;
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

alphas = logspace(-3,1,5);
gammas = logspace(-3,1,5);
betas = logspace(-3,1,5);
etas = 1;%[10 50 100];
mus = 1;%[10 100 1000];

delta1 = 1e-3; %1e-4
delta2 = 1e-3; %1e-4
epsilon = 1e-6;
model = 'lowrank';
err = zeros(length(mus),length(etas),length(betas),length(gammas),...
    length(alphas),n_graphs);
tic
parfor g=1:n_graphs
    disp(['Graph: ' num2str(g)])
    A = generate_connected_ER(N,p);
    As = gen_similar_graphs(A,K,pert_links);
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
    
    % Create covariances
    Cs = create_cov(As,L,M,sampled);
      
    Ao = As(n_o,n_o,:);
    Co = Cs(n_o,n_o,:);
    
    regs = struct('delta1',delta1,'delta2',delta2,'epsilon',epsilon);
    err_g = zeros(K,length(etas),length(mus),length(betas),...
        length(gammas),length(alphas));
    for i=1:length(alphas)
        regs.alpha = alphas(i);
        for j=1:length(gammas)
            regs.gamma = gammas(j);
            for k=1:length(betas)
                regs.beta = betas(k);
                for m=1:length(mus)
                    regs.mu = mus(m);
                    for l=1:length(etas)
                        regs.eta = etas(l);
                    
                        if strcmp(model,'baseline')
                            [Ao_hat,~] = estA_baseline(Co,regs);
                        elseif strcmp(model,'lowrank')
                            [Ao_hat,~] = estA_lowrank(Co,regs);
                        elseif strcmp(model,'lowrank rw')
                            [Ao_hat,~] = estA_lowrank_rw(Co,regs);
                        elseif strcmp(model,'grouplasso')
                            [Ao_hat,~] = estA_grouplasso(Co,regs);
                        elseif strcmp(model,'grouplasso rw')
                            [Ao_hat,~] = estA_grouplasso_rw(Co,regs);
                        end
                        Ao_hat = Ao_hat./max(max(Ao_hat));
                        diff_Ao = Ao-Ao_hat;
                        for n=1:K
                            norm_Ao = norm(Ao(:,:,n),'fro')^2;
                            err_g(n,l,m,k,j,i) = norm(diff_Ao(:,:,n),'fro')^2/norm_Ao;
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


figure()
err_aux = squeeze(median_err);
for a = 1:5
    subplot(3,3,a)
    imagesc(err_aux(:,:,a))
    colorbar()
    title(['alpha:' num2str(alphas(a))])
end
