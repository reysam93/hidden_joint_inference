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

alphas = logspace(-5,-1,25);
betas =  logspace(-5,-1,25);

err_lvgl = zeros(length(alphas),length(betas),n_graphs);
err_ggl = zeros(length(alphas),length(betas),n_graphs);
err_fgl = zeros(length(alphas),length(betas),n_graphs);
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
    Co = Cs(n_o,n_o,:);
    
    regs = struct();
    err_lvgl_g = zeros(K,length(alphas),length(betas));
    err_ggl_g = zeros(K,length(alphas),length(betas));
    err_fgl_g = zeros(K,length(alphas),length(betas));
    for i=1:length(betas)
        regs.beta = betas(i);
        regs.lambda2 = betas(i);
        for j=1:length(alphas)
            regs.alpha = alphas(j);
            regs.lambda1 = alphas(j);

            % Group Graphical Lasso
            Ao_ggl = gglasso(Co,regs);
            Ao_ggl = Ao_ggl./max(max(Ao_ggl));

            % Fused Graphical Lasso
            Ao_fgl = fglasso(Co,regs);
            Ao_fgl = Ao_fgl./max(max(Ao_fgl));

            % Latent Variable Graphical Lasso
            for k=1:K
                [Ao_lvgl,~] = LVGLASSO(Co(:,:,k),regs,false);
                Ao_lvgl = Ao_lvgl./max(max(Ao_lvgl));

                norm_Ao = norm(Ao(:,:,k),'fro')^2;
                err_lvgl_g(k,j,i) = norm(Ao(:,:,k)-Ao_lvgl,'fro')^2/norm_Ao;
                err_ggl_g(k,j,i) = norm(Ao(:,:,k)-Ao_ggl(:,:,k),'fro')^2/norm_Ao;
                err_fgl_g(k,j,i) = norm(Ao(:,:,k)-Ao_fgl(:,:,k),'fro')^2/norm_Ao;
            end

        end
    end
    err_lvgl(:,:,g) = mean(err_lvgl_g,1);
    err_ggl(:,:,g) = mean(err_ggl_g,1);
    err_fgl(:,:,g) = mean(err_fgl_g,1);
end
t = toc;
disp(['--- ' num2str(t/3600) ' hours'])
%% Display results
mean_err_lvgl = mean(err_lvgl,3);
mean_err_ggl = mean(err_ggl,3);
mean_err_fgl = mean(err_fgl,3);

disp(['Min mean err LV-GL: ' num2str(min(mean_err_lvgl(:)))])
disp(['Min mean err GGL: ' num2str(min(mean_err_ggl(:)))])
disp(['Min mean err FGL: ' num2str(min(mean_err_fgl(:)))])

figure()
imagesc(mean_err_lvgl)
colorbar()
xlabel('Alphas')
set(gca,'XTick',1:length(alphas))
set(gca,'XTickLabel',alphas)
ylabel('Betas')
set(gca,'YTick',1:length(betas))
set(gca,'YTickLabel',betas)
title('LV-GL')

figure()
imagesc(mean_err_ggl)
colorbar()
xlabel('Lambda1')
set(gca,'XTick',1:length(alphas))
set(gca,'XTickLabel',alphas)
ylabel('Lambda2')
set(gca,'YTick',1:length(betas))
set(gca,'YTickLabel',betas)
title('GGL')

figure()
imagesc(mean_err_fgl)
colorbar()
xlabel('Lambda1')
set(gca,'XTick',1:length(alphas))
set(gca,'XTickLabel',alphas)
ylabel('Lambda2')
set(gca,'YTick',1:length(betas))
set(gca,'YTickLabel',betas)
title('FGL')
