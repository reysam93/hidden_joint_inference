%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

n_graphs = 30;
sig_trials = 10;

K = 3;
N = 20;
O = 19;
p = 0.2;
F = 3;
pert_links = 3;
Ms = [1e2, 1e3, 1e4, 1e5]; %round(logspace(2,6,9));
hid_nodes = 'min';
max_iters = 10;
verb_freq = 20;

leg = {'LVGL,$C_{mrf}$','GGL,$C_{mrf}$','FGL,$C_{mrf}$','PGL,$C_{mrf}$',...
    'LVGL,$C_{poly}$','GGL,$C_{poly}$','FGL,$C_{poly}$','PGL,$C_{poly}$'};

regs_lvgl_mrf = struct();
regs_lvgl_mrf.alpha = 1e-2;
regs_lvgl_mrf.beta = 1e-3;

regs_mrf = struct();
regs_mrf.alpha   = 1;       % Sparsity of S
regs_mrf.gamma   = 1e4;      % Group Lasso
regs_mrf.beta    = 5;      % Similarity of S
regs_mrf.eta     = 10;      % Similarity of P
regs_mrf.mu      = 1e6;    % Commutative penalty
regs_mrf.delta1  = 1e-3;    % Small number for reweighted

regs_lvgl_poly = struct();
regs_lvgl_poly.alpha = 5e-3;
regs_lvgl_poly.beta = 5e-3;

% OLD REGS
regs_poly = struct();
regs_poly.alpha   = 1;       % Sparsity of S
regs_poly.gamma   = 100;      % Group Lasso
regs_poly.beta    = 5;      % Similarity of S
regs_poly.eta     = 5;      % Similarity of P
regs_poly.mu      = 1e3;    % Commutative penalty
regs_poly.delta1  = 1e-3;    % Small number for reweighted

regs_ggl_mrf = struct();
regs_ggl_mrf.lambda1 = 1e-3;
regs_ggl_mrf.lambda2 = 1e-3;

regs_ggl_poly = struct();
regs_ggl_poly.lambda1 = 1e-1;
regs_ggl_poly.lambda2 = 1e-1;

max_M = Ms(end);

tic
Aos = zeros(O,O,K,n_graphs);
P_mrf = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_lvgl_mrf = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_pgl_mrf = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_lvgl_poly = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_pgl_poly = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_ggl_mrf = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_fgl_mrf = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_ggl_poly = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_fgl_poly = zeros(O,O,K,length(Ms),sig_trials,n_graphs);

parfor g=1:n_graphs    
    % Create graphs and get hidden nodes
    A = generate_connected_ER(N,p);
    As = gen_similar_graphs(A,K,pert_links);
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
    Ao = As(n_o,n_o,:);
    
    % True Cmrf
    Cs_mrf_true = create_cov(As,F,inf,false,'mrf');
    Cs_poly_true = create_cov(As,F,inf,false,'poly');
    
%     % Graph filter (for C poly)
%     H = zeros(N,N,K);
%     for k=1:K
%         % h = rand(F,1)*2-1;
%         h = rand(F,1);
%         h = h/norm(h,1);
%         for f=1:F
%             H(:,:,k) = H(:,:,k) + h(f)*As(:,:,k)^(f-1);
%         end
%     end
    
    P_mrf_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_lvgl_mrf_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_pgl_mrf_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_lvgl_poly_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_pgl_poly_g = zeros(O,O,K,length(Ms),sig_trials);

    Aos_ggl_mrf_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_fgl_mrf_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_ggl_poly_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_fgl_poly_g = zeros(O,O,K,length(Ms),sig_trials);
    for j=1:sig_trials
        % Generate signals X
        X_mrf = zeros(N,max_M,K);
        X_poly = zeros(N,max_M,K);
        for k=1:K
            W = randn(N,max_M);
            X_mrf(:,:,k) = sqrtm(Cs_mrf_true(:,:,k))*W;
            X_poly(:,:,k) = sqrtm(Cs_poly_true(:,:,k))*W;
            %X_poly(:,:,k) = H(:,:,k)*W;
        end
        
        for i=1:length(Ms)
            M = Ms(i);
            
            % Compute covariance
            Cs_mrf = zeros(N,N,K);
            Cs_poly = zeros(N,N,K);
            for k=1:K
                Cs_mrf(:,:,k) = X_mrf(:,1:M,k)*X_mrf(:,1:M,k)'/M;
                Cs_poly(:,:,k) = X_poly(:,1:M,k)*X_poly(:,1:M,k)'/M;
            end
            Co_mrf = Cs_mrf(n_o,n_o,:);
            Co_poly = Cs_poly(n_o,n_o,:);
            
            %%%% Estimates of LatentVariable-GL %%%%
            for k=1:K
                % With C mrf
                Ao_hat = LVGLASSO(Co_mrf(:,:,k),regs_lvgl_mrf,false);
                Aos_lvgl_mrf_g(:,:,k,i,j) = Ao_hat./max(max(Ao_hat));
                
                % With C poly
                Ao_hat = LVGLASSO(Co_poly(:,:,k),regs_lvgl_poly,false);
                Aos_lvgl_poly_g(:,:,k,i,j) = Ao_hat./max(max(Ao_hat));
            end
            
            %%%% Estimates of Group-GL %%%%
            % With mrf
            Ao_hat = gglasso(Co_mrf,regs_ggl_mrf);
            Aos_ggl_mrf_g(:,:,:,i,j) = Ao_hat./max(max(Ao_hat));

            % With C poly
            Ao_hat = gglasso(Co_poly,regs_ggl_poly);
            Aos_ggl_poly_g(:,:,:,i,j) = Ao_hat./max(max(Ao_hat));

            %%%% Estimates of Fusion-GL %%%%
            % With mrf
            Ao_hat = fglasso(Co_mrf,regs_ggl_mrf);
            Aos_fgl_mrf_g(:,:,:,i,j) = Ao_hat./max(max(Ao_hat));

            % With C poly
            Ao_hat = fglasso(Co_poly,regs_ggl_poly);
            Aos_fgl_poly_g(:,:,:,i,j) = Ao_hat./max(max(Ao_hat));

            %%%% Estimates of Pgl %%%%
            % With mrf
            Ao_hat = PGL_rw(Co_mrf,regs_mrf,max_iters);
            Aos_pgl_mrf_g(:,:,:,i,j) = Ao_hat./max(max(Ao_hat));
            
            % With C poly
            [Ao_hat,~] = PGL_rw(Co_poly,regs_poly,max_iters);
            Aos_pgl_poly_g(:,:,:,i,j) = Ao_hat./max(max(Ao_hat));

            if mod(g,verb_freq) == 1
                disp(['G: ' num2str(g) ' M: ' num2str(M) ' Sig trial: '...
                      num2str(sig_trials)])
            end
        end
    end
    Aos(:,:,:,g) = Ao;
    P_mrf(:,:,:,:,:,g) = P_mrf_g;
    Aos_lvgl_mrf(:,:,:,:,:,g) = Aos_lvgl_mrf_g;
    Aos_pgl_mrf(:,:,:,:,:,g) = Aos_pgl_mrf_g;
    Aos_lvgl_poly(:,:,:,:,:,g) = Aos_lvgl_poly_g;
    Aos_pgl_poly(:,:,:,:,:,g) = Aos_pgl_poly_g;
    Aos_ggl_mrf(:,:,:,:,:,g) = Aos_ggl_mrf_g;
    Aos_fgl_mrf(:,:,:,:,:,g) = Aos_fgl_mrf_g;
    Aos_ggl_poly(:,:,:,:,:,g) = Aos_ggl_poly_g;
    Aos_fgl_poly(:,:,:,:,:,g) = Aos_fgl_poly_g;
end
t = toc;
disp(['----- ' num2str(t/3600) ' hours -----'])

%% Plot mean/median error
err = zeros(K,length(Ms),length(leg),sig_trials,n_graphs);
for g=1:n_graphs
    for j=1:sig_trials
        for k=1:K
            norm_A = norm(Aos(:,:,k,g),'fro')^2;
            for i=1:length(Ms)
                % Matrices normalized to maximum value 1
                err(k,i,1,j,g) = ...
                    norm(Aos(:,:,k,g)-Aos_lvgl_mrf(:,:,k,i,j,g),'fro')^2/norm_A;
                err(k,i,2,j,g) = ...
                    norm(Aos(:,:,k,g)-Aos_ggl_mrf(:,:,k,i,j,g),'fro')^2/norm_A;
                err(k,i,3,j,g) = ...
                    norm(Aos(:,:,k,g)-Aos_fgl_mrf(:,:,k,i,j,g),'fro')^2/norm_A;
                err(k,i,4,j,g) = ...
                    norm(Aos(:,:,k,g)-Aos_pgl_mrf(:,:,k,i,j,g),'fro')^2/norm_A;
                err(k,i,5,j,g) = ...
                    norm(Aos(:,:,k,g)-Aos_lvgl_poly(:,:,k,i,j,g),'fro')^2/norm_A;
                err(k,i,6,j,g) = ...
                    norm(Aos(:,:,k,g)-Aos_ggl_poly(:,:,k,i,j,g),'fro')^2/norm_A;
                err(k,i,7,j,g) = ...
                    norm(Aos(:,:,k,g)-Aos_fgl_poly(:,:,k,i,j,g),'fro')^2/norm_A;
                err(k,i,8,j,g) = ...
                    norm(Aos(:,:,k,g)-Aos_pgl_poly(:,:,k,i,j,g),'fro')^2/norm_A;
            end
        end
    end
end

mean_err = squeeze(mean(mean(mean(err,1),4),5));

% Plot properties
mark_s = 8;
line_w = 2;

%leg(5) = [];

figure();
semilogx(Ms,mean_err(:,1),':x','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_err(:,2),':v','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_err(:,3),':s','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_err(:,4),':o','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_err(:,5),'-x','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_err(:,6),'-v','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_err(:,7),'-s','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_err(:,8),'-o','LineWidth',line_w,'MarkerSize',mark_s); hold off
xlabel('(b) Number of samples')
ylabel('Mean error')
legend(leg,'Location','east')
grid on
% ylim([0 1.5])
set(gca,'FontSize',16);
set(gcf, 'PaperPositionMode', 'auto')
