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
Cmrf = false;

alphas = logspace(-5,-1,20);
betas = logspace(-5,-1,20);

err = zeros(length(alphas),length(betas),n_graphs);
tic
for g=1:n_graphs
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
    Co = Cs(n_o,n_o,:);
    
    regs = struct();
    err_g = zeros(K,length(alphas),length(betas));
    for i=1:length(betas)
        regs.beta = betas(i);
        for j=1:length(alphas)
            regs.alpha = alphas(j);
            for k=1:K
                [Ao_hat,~] = LVGLASSO(Co(:,:,k),regs,false);
                Ao_hat = Ao_hat./max(max(Ao_hat));

                norm_Ao = norm(Ao(:,:,k),'fro')^2;
                err_g(k,j,i) = norm(Ao(:,:,k)-Ao_hat,'fro')^2/norm_Ao;
            end
        end
    end
    err(:,:,g) = mean(err_g,1);
end
t = toc;
disp(['--- ' num2str(t/3600) ' hours'])
%% Display results
mean_err = mean(err,3);
median_err = median(err,3);

disp(['Min mean err: ' num2str(min(mean_err(:)))])
disp(['Min median err: ' num2str(min(median_err(:)))])

figure()
imagesc(mean_err)
colorbar()
xlabel('Alphas')
set(gca,'XTick',1:length(alphas))
set(gca,'XTickLabel',alphas)
ylabel('Betas')
set(gca,'YTick',1:length(betas))
set(gca,'YTickLabel',betas)
