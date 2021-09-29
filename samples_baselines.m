%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

n_graphs = 24;
sig_trials = 1; %5;

K = 3;
N = 20;
O = 19;
p = 0.2;
F = 3;
pert_links = 3;
Ms = [1e3 1e4] %round(logspace(2,5,4));
hid_nodes = 'min';
max_iters = 10;
th = 0.3;

leg = {'LVGL,C_{mrf}','Pgl,C_{mrf}','LVGL,C_{poly}','Pgl,C_{poly}'};

regs_lvgl_mrf = struct();
regs_lvgl_mrf.alpha = 1e-3;
regs_lvgl_mrf.beta = 1e-3;

regs_mrf = struct();
regs_mrf.alpha   = 1e-2;       % Sparsity of S
regs_mrf.gamma   = 1e2;      % Group Lasso
regs_mrf.beta    = 0;      % Similarity of S
regs_mrf.eta     = 0;      % Similarity of P
regs_mrf.mu      = 1e3;    % Commutative penalty
regs_mrf.delta1  = 1e-3;    % Small number for reweighted

regs_lvgl_poly = struct();
regs_lvgl_poly.alpha = 1; %1e-5;
regs_lvgl_poly.beta = 1e-3; %1e-2;

regs_poly = struct();
regs_poly.alpha   = 1e-3;       % Sparsity of S
regs_poly.gamma   = 25;      % Group Lasso
regs_poly.beta    = 1;      % Similarity of S
regs_poly.eta     = 25;      % Similarity of P
regs_poly.mu      = 1000;    % Commutative penalty
regs_poly.delta1  = 1e-3;    % Small number for reweighted

max_M = Ms(end);

tic
Aos = zeros(O,O,K,n_graphs);
Aos_lvgl_mrf = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_pgl_mrf = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_lvgl_poly = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_pgl_poly = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
for g=1:n_graphs
    disp(['G: ' num2str(g)])
    
    % Create graphs and get hidden nodes
    A = generate_connected_ER(N,p);
    As = gen_similar_graphs(A,K,pert_links);
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
    Ao = As(n_o,n_o,:);
    Aoh = As(n_o,n_h,:);
    
    % True Cmrf
    Cs_mrf_true = zeros(N,N,K);
    % figure();
    for k=1:K
        eigvals = eig(As(:,:,k));
        C_inv = (0.01-min(eigvals))*eye(N,N) + (0.9+0.1*rand(1,1))*As(:,:,k);
        Cs_mrf_true(:,:,k) = inv(C_inv);
        Dc = eig(Cs_mrf_true(:,:,k));
        assert(min(Dc)>=0)
    end
    
    % Graph filter (for C poly)
    H = zeros(N,N,K);
    for k=1:K
        h = rand(F,1)*2-1;
        for f=1:F
            H(:,:,k) = H(:,:,k) + h(f)*As(:,:,k)^(f-1);
        end
    end
    
    Aos_lvgl_mrf_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_pgl_mrf_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_lvgl_poly_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_pgl_poly_g = zeros(O,O,K,length(Ms),sig_trials);
    for j=1:sig_trials
        % Generate signals X
        X_poly = zeros(N,max_M,K);
        X_mrf = zeros(N,max_M,K);
        for k=1:K
            W = randn(N,max_M);
            X_poly(:,:,k) = H(:,:,k)*W;
            X_mrf(:,:,k) = sqrtm(Cs_mrf_true(:,:,k))*W;
        end
        
        for i=1:length(Ms)
            M = Ms(i);
            disp(['   M: ' num2str(M)])
            
            % Compute covariance
            Cs_poly = zeros(N,N,K);
            Cs_mrf = zeros(N,N,K);
            for k=1:K
                Cs_mrf(:,:,k) = X_mrf(:,1:M,k)*X_mrf(:,1:M,k)'/M;
                Cs_poly(:,:,k) = X_poly(:,1:M,k)*X_poly(:,1:M,k)'/M;
                disp(['- k: ' num2str(k) ' norm Ck mrf '...
                num2str(norm(Cs_mrf(:,:,k),'fro')) ' norm Ck poly '...
                num2str(norm(Cs_poly(:,:,k),'fro'))])
            end
            Co_poly = Cs_poly(n_o,n_o,:);
            Co_mrf = Cs_mrf(n_o,n_o,:);
            
            %%%% Estimates of LatentVariable-GL %%%%
            for k=1:K
                % With C mrf
                Ao_hat = LVGLASSO(Co_mrf(:,:,k),regs_lvgl_mrf,false);
                Aos_lvgl_mrf_g(:,:,k,i,j) = Ao_hat./max(max(Ao_hat));
                
                % With C poly
                Ao_hat = LVGLASSO(Co_poly(:,:,k),regs_lvgl_poly,false);
                Aos_lvgl_poly_g(:,:,k,i,j) = Ao_hat./max(max(Ao_hat));
            end
            
            %%%% Estimates of Pgl %%%%
            % With mrf
            [Ao_hat,~] = estA_pgl_colsp_rw2(Co_mrf,N-O,regs_mrf,max_iters,true);
            Aos_pgl_mrf_g(:,:,:,i,j) = Ao_hat./max(max(Ao_hat));

            % With C poly
            [Ao_hat,~] = estA_pgl_colsp_rw2(Co_poly,N-O,regs_poly,max_iters);
            Aos_pgl_poly_g(:,:,:,i,j) = Ao_hat./max(max(Ao_hat));
        end
    end
    Aos(:,:,:,g) = Ao;
    Aos_lvgl_mrf(:,:,:,:,:,g) = Aos_lvgl_mrf_g;
    Aos_pgl_mrf(:,:,:,:,:,g) = Aos_pgl_mrf_g;
    Aos_lvgl_poly(:,:,:,:,:,g) = Aos_lvgl_poly_g;
    Aos_pgl_poly(:,:,:,:,:,g) = Aos_pgl_poly_g;
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
                err(k,i,1,g) = ...
                    norm(Ao(:,:,k,g)-Aos_lvgl_mrf(:,:,k,i,j,g),'fro')^2/norm_A;
                err(k,i,2,g) = ...
                    norm(Ao(:,:,k,g)-Aos_pgl_mrf(:,:,k,i,j,g),'fro')^2/norm_A;
                err(k,i,3,g) = ...
                    norm(Ao(:,:,k,g)-Aos_lvgl_poly(:,:,k,i,j,g),'fro')^2/norm_A;
                err(k,i,4,g) = ...
                    norm(Ao(:,:,k,g)-Aos_pgl_poly(:,:,k,i,j,g),'fro')^2/norm_A;
            end
        end
    end
end

mean_err = squeeze(mean(mean(mean(err,1),4),5));
med_err = squeeze(median(median(mean(err,1),4),5));

% rec_joint = sum(sum(err <= .1,4),5)/(n_graphs*sig_trials);
% rec_sep= sum(sum(err <= .1,4),5)/(n_graphs*sig_trials);

% Mean error
figure();
semilogx(Ms,mean_err(:,1),'-x'); hold on
semilogx(Ms,mean_err(:,2),'-o'); hold on
semilogx(Ms,mean_err(:,3),'--x'); hold on
semilogx(Ms,mean_err(:,4),'--o'); hold off
xlabel('Number of samples')
ylabel('Mean error')
legend(leg)
grid on; axis tight

% % Median error
% figure();
% semilogx(Ms,med_err_joint(1,:),'-o'); hold on
% semilogx(Ms,med_err_joint(2,:),'-x'); hold on
% semilogx(Ms,med_err_joint(3,:),'-v'); hold on
% semilogx(Ms,med_err_sep(1,:),'--o'); hold on
% semilogx(Ms,med_err_sep(2,:),'--x'); hold on
% semilogx(Ms,med_err_sep(3,:),'--v'); hold off
% xlabel('Number of samples')
% ylabel('Median error')
% legend(leg)
% grid on; axis tight

% % Median error
% figure();
% semilogx(Ms,rec_joint(1,:),'-o'); hold on
% semilogx(Ms,rec_joint(2,:),'-x'); hold on
% semilogx(Ms,rec_joint(3,:),'-v'); hold on
% semilogx(Ms,rec_sep(1,:),'--o'); hold on
% semilogx(Ms,rec_sep(2,:),'--x'); hold on
% semilogx(Ms,rec_sep(3,:),'--v'); hold off
% xlabel('Number of samples')
% ylabel('Recovered graphs (err)')
% legend(leg)
% grid on; axis tight

%% Plot fscore and recovered graphs
% fsc_joint = zeros(K,length(Ms),sig_trials,n_graphs);
% fsc_sep = zeros(K,length(Ms),sig_trials,n_graphs);
% for g=1:n_graphs
%     Ao_th = Aos(:,:,k,g);
%     Ao_th(Ao_th >= th) = 1;
%     Ao_th(Ao_th < th) = 0;
%     
%     for j=1:sig_trials
%         for i=1:length(Ms)
%             for k=1:K            
%                 Ao_j_th = Aos_joint(:,:,k,i,j,g);
%                 Ao_j_th(Ao_j_th >= th) = 1;
%                 Ao_j_th(Ao_j_th < th) = 0;
%                 Ao_s_th = Aos_sep(:,:,k,i,j,g);
%                 Ao_s_th(Ao_s_th >= th) = 1;
%                 Ao_s_th(Ao_s_th < th) = 0;
%                 
%                 [~,~,fsc_joint(k,i,j,g),~,~] = ...
%                     graph_learning_perf_eval(Ao_th,Ao_j_th);
%                 [~,~,fsc_sep(k,i,j,g),~,~] = ...
%                     graph_learning_perf_eval(Ao_th,Ao_s_th);
%             end
%         end
%     end
% end
% 
% mean_fsc_joint = squeeze(mean(mean(fsc_joint,3),4));
% mean_fsc_sep = squeeze(mean(mean(fsc_sep,3),4));
% med_fsc_joint = squeeze(median(median(fsc_joint,3),4));
% med_fsc_sep = squeeze(median(median(fsc_joint,3),4));
% 
% rec_joint = sum(sum(fsc_joint == 1,3),4)/(n_graphs*sig_trials);
% rec_sep= sum(sum(fsc_sep == 1,3),4)/(n_graphs*sig_trials);
% 
% % Mean fsc
% figure();
% semilogx(Ms,mean_fsc_joint(1,:),'-o'); hold on
% semilogx(Ms,mean_fsc_joint(2,:),'-x'); hold on
% semilogx(Ms,mean_fsc_joint(3,:),'-v'); hold on
% semilogx(Ms,mean_fsc_sep(1,:),'--o'); hold on
% semilogx(Ms,mean_fsc_sep(2,:),'--x'); hold on
% semilogx(Ms,mean_fsc_sep(3,:),'--v'); hold off
% xlabel('Number of samples')
% ylabel('Mean fscore')
% legend(leg)
% grid on; axis tight
% 
% % Median fsc
% figure();
% semilogx(Ms,med_fsc_joint(1,:),'-o'); hold on
% semilogx(Ms,med_fsc_joint(2,:),'-x'); hold on
% semilogx(Ms,med_fsc_joint(3,:),'-v'); hold on
% semilogx(Ms,med_fsc_sep(1,:),'--o'); hold on
% semilogx(Ms,med_fsc_sep(2,:),'--x'); hold on
% semilogx(Ms,med_fsc_sep(3,:),'--v'); hold off
% xlabel('Number of samples')
% ylabel('Median fscore')
% legend(leg)
% grid on; axis tight
% 
% % Recovered graphs
% figure();
% semilogx(Ms,rec_joint(1,:),'-o'); hold on
% semilogx(Ms,rec_joint(2,:),'-x'); hold on
% semilogx(Ms,rec_joint(3,:),'-v'); hold on
% semilogx(Ms,rec_sep(1,:),'--o'); hold on
% semilogx(Ms,rec_sep(2,:),'--x'); hold on
% semilogx(Ms,rec_sep(3,:),'--v'); hold off
% xlabel('Number of samples')
% ylabel('Recovered graphs (fsc)')
% legend(leg)
% grid on; axis tight