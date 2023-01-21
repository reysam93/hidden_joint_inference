rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

clear

% Exp parameters
nG = 30;
p_rew_links = [0, .05, .1, .15, .2, .25, .3];
K = 4;
N = 20;
O = 19;
H = N-O;
p = 0.2;
L = 3;
M = 1e4;
sampled = true;
hid_nodes = 'min';
max_iters = 10;
verb_freq = 20;

% REGS LVG
regs_lvgl_mrf = struct();
regs_lvgl_mrf.alpha = 1e-2;
regs_lvgl_mrf.beta = 1e-3;

regs_lvgl_poly = struct();
regs_lvgl_poly.alpha = 5e-3;
regs_lvgl_poly.beta = 5e-3;

% REGS FGL
regs_fgl_mrf = struct();
regs_fgl_mrf.lambda1 = 1e-3;
regs_fgl_mrf.lambda2 = 1e-3;

regs_fgl_poly = struct();
regs_fgl_poly.lambda1 = 1e-1;
regs_fgl_poly.lambda2 = 1e-1;


% REGS PGL
% NEED ADJUSTING
regs_mrf = struct();
regs_mrf.alpha   = 1;       % Sparsity of S
regs_mrf.gamma   = 1e4;      % Group Lasso
regs_mrf.beta    = 5;      % Similarity of S
regs_mrf.eta     = 10;      % Similarity of P
regs_mrf.mu      = 1e6;    % Commutative penalty
regs_mrf.delta1  = 1e-3;    % Small number for reweighted

regs_poly = struct();
regs_poly.alpha   = 1;      % Sparsity of S
regs_poly.gamma   = 100;    % Group Lasso (each P)
regs_poly.beta    = 10;      % Similarity of Ss
regs_poly.eta     = 10;      % Similarity of Ps
regs_poly.mu      = 1e4;    % Commutative penalty
regs_poly.delta1  = 1e-3;   % Small number for reweighted

% LVGL, FGL, Our, Ours GMRF
models = {'LVGL,C_{poly}','FGL,C_{poly}','PGL,C_{poly}',...
    'LVGL,C_{MRF}','FGL,C_{MRF}','PGL,C_{MRF}'};

%%
err1 = zeros(length(models),length(p_rew_links),nG);
err2 = zeros(length(models),length(p_rew_links),nG);
tic
for g = 1:nG
    A = generate_connected_ER(N,p);
    [n_o, n_h] = select_hidden_nodes(hid_nodes,O,A);

    % Filter coeffs h are independent of graph similarity
    hs = zeros(L,K);
    for k=1:K
        hs(:,k) = rand(L,1);
        hs(:,k) = hs(:,k)/norm(hs(:,k),1);
    end

    err1_g = zeros(length(models),length(p_rew_links));
    err2_g = zeros(length(models),length(p_rew_links));
    for i = 1:length(p_rew_links)
        n_rew_links = round(sum(sum(A))/2*p_rew_links(i));

        As = gen_similar_graphs(A,K,n_rew_links);
        Ao = As(n_o,n_o,:);

        % Create covariances
        Cs_mrf = create_cov(As,L,M,sampled,'mrf');
        Cs_poly = create_cov(As,L,M,sampled,'poly',hs);
        Co_mrf = Cs_mrf(n_o,n_o,:);
        Co_poly = Cs_poly (n_o,n_o,:);

        %%%% Estimates of LatentVariable-GL %%%%
        A_lvgl_mrf = zeros(O,O,K);
        A_lvgl_poly = zeros(O,O,K);
        for k=1:K
            % With C mrf
            A_lvgl_mrf(:,:,k) = LVGLASSO(Co_mrf(:,:,k),regs_lvgl_mrf,false);
            A_lvgl_mrf(:,:,k) = A_lvgl_mrf(:,:,k)./max(max(A_lvgl_mrf(:,:,k)));

            % With C poly
            A_lvgl_poly(:,:,k) = LVGLASSO(Co_poly(:,:,k),regs_lvgl_poly,false);
            A_lvgl_poly(:,:,k) = A_lvgl_poly(:,:,k)./max(max(A_lvgl_poly(:,:,k)));
        end


        %%%% Estimates of Fusion-GL %%%%
        % With mrf
        A_fgl_mrf = fglasso(Co_mrf,regs_fgl_mrf);
        A_fgl_mrf = A_fgl_mrf./max(max(A_fgl_mrf));

        % With C poly
        A_fgl_poly = fglasso(Co_poly,regs_fgl_poly);
        A_fgl_poly = A_fgl_poly./max(max(A_fgl_poly));


        %%%% Estimates J-LVGL %%%%
        % PENDING

        %%%% Estimates of Pgl %%%%
        % With C mrf
        A_pgl_mrf = PGL_rw(Co_mrf,regs_mrf,max_iters);
        A_pgl_mrf = A_pgl_mrf./max(max(A_pgl_mrf));

        % With C poly
        [A_pgl_poly,~] = PGL_rw(Co_poly,regs_poly,max_iters);
        A_pgl_poly = A_pgl_poly./max(max(A_pgl_poly));

       % Compute error
        for k = 1:K
            Aok = Ao(:,:,k);
            norm_Aok = norm(Aok,'fro');
            Aok_n = Aok/norm_Aok;

            err1_g(1,i) = err1_g(1,i) + (norm(Aok-A_lvgl_poly(:,:,k),'fro')/norm_Aok)^2/K;
            err1_g(2,i) = err1_g(2,i) + (norm(Aok-A_fgl_poly(:,:,k),'fro')/norm_Aok)^2/K;
            err1_g(3,i) = err1_g(3,i) + (norm(Aok-A_pgl_poly(:,:,k),'fro')/norm_Aok)^2/K;
            err1_g(4,i) = err1_g(4,i) + (norm(Aok-A_lvgl_mrf(:,:,k),'fro')/norm_Aok)^2/K;
            err1_g(5,i) = err1_g(5,i) + (norm(Aok-A_fgl_mrf(:,:,k),'fro')/norm_Aok)^2/K;
            err1_g(6,i) = err1_g(6,i) + (norm(Aok-A_pgl_mrf(:,:,k),'fro')/norm_Aok)^2/K;

            A_lvgl_poly_n = A_lvgl_poly(:,:,k)/norm(A_lvgl_poly(:,:,k),'fro');
            A_fgl_poly_n = A_fgl_poly(:,:,k)/norm(A_fgl_poly(:,:,k),'fro');
            A_pgl_poly_n = A_pgl_poly(:,:,k)/norm(A_pgl_poly(:,:,k),'fro');
            A_lvgl_mrf_n = A_lvgl_mrf(:,:,k)/norm(A_lvgl_mrf(:,:,k),'fro');
            A_fgl_mrf_n = A_fgl_mrf(:,:,k)/norm(A_fgl_mrf(:,:,k),'fro');
            A_pgl_mrf_n = A_pgl_mrf(:,:,k)/norm(A_pgl_mrf(:,:,k),'fro');

            err2_g(1,i) = err2_g(1,i) + norm(Aok_n-A_lvgl_poly_n,'fro')^2/K;
            err2_g(2,i) = err2_g(2,i) + norm(Aok_n-A_fgl_poly_n,'fro')^2/K;
            err2_g(3,i) = err2_g(3,i) + norm(Aok_n-A_pgl_poly_n,'fro')^2/K;
            err2_g(4,i) = err2_g(4,i) + norm(Aok_n-A_lvgl_mrf_n,'fro')^2/K;
            err2_g(5,i) = err2_g(5,i) + norm(Aok_n-A_fgl_mrf_n,'fro')^2/K;
            err2_g(6,i) = err2_g(6,i) + norm(Aok_n-A_pgl_mrf_n,'fro')^2/K;
        end

        if mod(g,verb_freq) == 1
            disp(['Graph: ' num2str(g) ' Rew links: ' num2str(n_rew_links) ' Err: ' num2str(err1_g(3,i))])
        end

    end
    err1(:,:,g) = err1_g;
    err2(:,:,g) = err2_g;
end
t = toc;
disp(['----- ' num2str(t/60) ' mins -----'])

%%
mean_err = mean(err1,3);
mean_err2 = mean(err2,3);

figure()
plot(p_rew_links,mean_err)
legend(models)
xlabel('Prop. diff. links')
ylim([0 1])

figure()
plot(p_rew_links,mean_err2)
legend(models)
xlabel('Prop. diff. links')
ylim([0 1])

%%
median_err = median(err1,3);
median_err2 = median(err2,3);

figure()
plot(p_rew_links,median_err)
legend(models)
xlabel('Prop. diff. links')
ylim([0 1])

figure()
plot(p_rew_links,median_err2)
legend(models)
xlabel('Prop. diff. links')
ylim([0 1])









