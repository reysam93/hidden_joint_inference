%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

%%%% SETTING %%%%%
REW_ONLY_OBS = false;
C_TYPE = 'poly';  % or mrf

n_graphs = 15;
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
verb_freq = 10;
th = 0.3;
Cmrf = true;

alphas = logspace(-5,-2,5);
betas =  logspace(-5,-1,9);

err_lvgl = zeros(length(alphas),length(betas),n_graphs);
err_ggl = zeros(length(alphas),length(betas),n_graphs);
err_fgl = zeros(length(alphas),length(betas),n_graphs);
tic
parfor g=1:n_graphs
    A = generate_connected_ER(N,p);
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, A);
    %if REW_ONLY_OBS
    %    As = gen_similar_graphs_hid(A,Ks(end),rew_links,n_o,n_h);
    %else
        As = gen_similar_graphs(A,K,rew_links);
    %end
    Cs = create_cov(As,F,M,sampled, C_TYPE);
    
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
            
            if mod(g,verb_freq) == 1
                disp(['Graph: ' num2str(g) ' beta: ' num2str(regs.beta)...
                    ' alphas: ' num2str(regs.alpha) ' err (LVGL): '...
                    num2str(err_lvgl_g(k,j,i)) ' err (FGL): ' ...
                    num2str(err_fgl_g(k,j,i))])
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
mean_err_fgl = mean(err_fgl,3);

[~, lin_idx] = min(mean_err_lvgl(:));
[j,i] = ind2sub(size(mean_err_lvgl),lin_idx);
disp(['Min err  LV-GL: Alpha: ' num2str(alphas(j)) ' Beta: ' num2str(betas(i))...
      ' Err: ' num2str(mean_err_lvgl(j,i))])
  
[~, lin_idx] = min(mean_err_fgl(:));
[j,i] = ind2sub(size(mean_err_fgl),lin_idx);
disp(['Min err  FGL: Lambda1: ' num2str(alphas(j)) ' Lambda2: ' num2str(betas(i))...
      ' Err: ' num2str(mean_err_fgl(j,i))])
  
  
disp(['Min mean err LV-GL: ' num2str(min(mean_err_lvgl(:)))])
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
imagesc(mean_err_fgl)
colorbar()
xlabel('Lambda1')
set(gca,'XTick',1:length(alphas))
set(gca,'XTickLabel',alphas)
ylabel('Lambda2')
set(gca,'YTick',1:length(betas))
set(gca,'YTickLabel',betas)
title('FGL')
