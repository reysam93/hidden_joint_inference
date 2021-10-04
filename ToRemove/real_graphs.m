%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

graphs = [7 9 12];
signal_trials = 1;
N = 32;
O = 31;
Ms = [1e4];
F = 4;
K = length(graphs);
n_models = 4;
max_M = Ms(end);
hid_nodes = 'min';
th = 0.3;

leg = {'LVGL,C_{mrf}','Pgl,C_{mrf}','LVGL,C_{poly}','Pgl,C_{poly}'};

max_iters = 10;
regs_mrf = struct();
regs_mrf.alpha   = 1e-3;       % Sparsity of S
regs_mrf.gamma   = 25;      % Group Lasso
regs_mrf.beta    = 1;      % Similarity of S
regs_mrf.eta     = 25;      % Similarity of P
regs_mrf.mu      = 100;    % Commutative penalty
regs_mrf.delta1  = 1e-3;    % Small number for reweighted

regs_poly = struct();
regs_poly.alpha   = 1;       % Sparsity of S
regs_poly.gamma   = 100;      % Group Lasso
regs_poly.beta    = 50;      % Similarity of S
regs_poly.eta     = 25;      % Similarity of P
regs_poly.mu      = 1000;    % Commutative penalty
regs_poly.delta1  = 1e-3;    % Small number for reweighted

regs_lvgl_mrf = struct();
regs_lvgl_mrf.alpha = 1e-3;
regs_lvgl_mrf.beta = 1e-3;

regs_lvgl_poly = struct();
regs_lvgl_poly.alpha = 1e-5;
regs_lvgl_poly.beta = 1e-2;

% Load graphs
As = get_student_networks_graphs(graphs,N);

% Try with whole graph also
[n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
Ao = As(n_o,n_o,:);

% True Cmrf
Cs_mrf_true = zeros(N,N,K);
% figure();
for k=1:K
    eigvals = eig(As(:,:,k));
    C_inv = (0.01-min(eigvals))*eye(N,N) + (0.9+0.1*rand(1,1))*As(:,:,k);
%     C_inv = 2.5*eye(N) + (0.9 + 0.1*rand(1,1))*As(:,:,k);
%     subplot(2,3,k);imagesc(C_inv);colorbar()
    
    Cs_mrf_true(:,:,k) = inv(C_inv);
%     subplot(2,3,k+K);imagesc(Cs_mrf_true(:,:,k));colorbar()
    Dc = eig(Cs_mrf_true(:,:,k));
    assert(min(Dc)>=0)        
end

% Graph filter (for C poly)
H = zeros(N,N,K);
for k=1:K
    h = rand(F,1)*2-1;
    for f=1:F
        H(:,:,k) = H(:,:,k) + h(f)*(As(:,:,k)^(f-1));
    end
end

tic
Aos_lvgl_mrf = zeros(O,O,K,length(Ms),signal_trials);
Aos_pgl_mrf = zeros(O,O,K,length(Ms),signal_trials);
Aos_lvgl_poly = zeros(O,O,K,length(Ms),signal_trials);
Aos_pgl_poly = zeros(O,O,K,length(Ms),signal_trials);
for g=1:signal_trials
    disp(['Trial: ' num2str(g)])
    
    % Generate signals X
    X_poly = zeros(N,max_M,K);
    X_mrf = zeros(N,max_M,K);
    for k=1:K
        W = randn(N,max_M);
        X_poly(:,:,k) = H(:,:,k)*W;
        X_mrf(:,:,k) = sqrtm(Cs_mrf_true(:,:,k))*W;
    end
    
    Aos_lvgl_mrf_g = zeros(O,O,K,length(Ms));
    Aos_pgl_mrf_g = zeros(O,O,K,length(Ms));
    Aos_lvgl_poly_g = zeros(O,O,K,length(Ms));
    Aos_pgl_poly_g = zeros(O,O,K,length(Ms));
    for i=1:length(Ms)
        M = Ms(i);
        
        % Compute covariance
        Cs_poly = zeros(N,N,K);
        Cs_mrf = zeros(N,N,K);
%         figure()
        for k=1:K
            Cs_mrf(:,:,k) = X_mrf(:,1:M,k)*X_mrf(:,1:M,k)'/M;
            Cs_poly(:,:,k) = X_poly(:,1:M,k)*X_poly(:,1:M,k)'/M;
            
%             Cs_mrf(:,:,k) = Cs_mrf_true(:,:,k);
%             Cs_poly(:,:,k) = H(:,:,k)*H(:,:,k)';
            
%             subplot(2,3,k);imagesc(Cs_mrf(:,:,k));colorbar()
%             subplot(2,3,k+K);imagesc(Cs_poly(:,:,k));colorbar()
            
%             Cs_mrf(:,:,k)=Cs_mrf(:,:,k)/norm(Cs_mrf(:,:,k),'fro')*...
%                 norm(Cs_poly(:,:,k),'fro');
            
            disp(['- k: ' num2str(k) ' norm Ck_mrf '...
                num2str(norm(Cs_mrf(:,:,k),'fro')) ' norm Ck_poly '...
                num2str(norm(Cs_poly(:,:,k),'fro'))])
            disp(['   - Comm mrf: '...
                num2str(norm(Cs_mrf(:,:,k)*As(:,:,k)-As(:,:,k)*Cs_mrf(:,:,k),'fro'))...
                ' Comm poly '...
                num2str(norm(Cs_poly(:,:,k)*As(:,:,k)-As(:,:,k)*Cs_poly(:,:,k),'fro'))])
        end
        Co_poly = Cs_poly(n_o,n_o,:);
        Co_mrf = Cs_mrf(n_o,n_o,:);        
        
        %%%% Estimates of LatentVariable-GL %%%%
        for k=1:K
            % With C mrf
%             Ao_hat = LVGLASSO(Co_mrf(:,:,k),regs_lvgl_mrf,false);
%             Aos_lvgl_mrf_g(:,:,k,i) = Ao_hat./max(max(Ao_hat));
%             
%             % With C poly
%             Ao_hat = LVGLASSO(Co_poly(:,:,k),regs_lvgl_poly,false);
%             Aos_lvgl_poly_g(:,:,k,i) = Ao_hat./max(max(Ao_hat));
        end
        
        %%%% Estimates of Pgl %%%%
        % With mrf
        [Ao_hat,P] = estA_pgl_colsp_rw2(Co_mrf,N-O,regs_mrf,max_iters,true);
        Aos_pgl_mrf_g(:,:,:,i) = Ao_hat./max(max(Ao_hat));
%         
%         % With C poly
%         [Ao_hat,P] = estA_pgl_colsp_rw2(Co_poly,N-O,regs,max_iters);
%         Aos_pgl_poly_g(:,:,:,i) = Ao_hat./max(max(Ao_hat));
        
        if g==1 ||g==2 || g==10
            figure()
            for k=1:K
                subplot(3,K,k)
                imagesc(Ao(:,:,k))
                colorbar()
                title(['True A: ' num2str(k)])
                subplot(3,K,k+K)
                imagesc(Aos_pgl_mrf_g(:,:,k,i))
                colorbar()
                title(['A: ' num2str(k)])
                subplot(3,K,k+2*K)
                imagesc(P(:,:,k))
                colorbar()
                title(['P: ' num2str(k)])
            end
        end
    end
    
    Aos_lvgl_mrf(:,:,:,:,g) = Aos_lvgl_mrf_g;
    Aos_pgl_mrf(:,:,:,:,g) = Aos_pgl_mrf_g;
    Aos_lvgl_poly(:,:,:,:,g) = Aos_lvgl_poly_g;
    Aos_pgl_poly(:,:,:,:,g) = Aos_pgl_poly_g;
end
t = toc;
disp(['----- ' num2str(t/3600) ' hours -----'])


%% Plot mean/median error
err = zeros(K,length(Ms),n_models,signal_trials);
for g=1:signal_trials
    for k=1:K
        norm_A = norm(Ao(:,:,k),'fro')^2;
        for i=1:length(Ms)
            err(k,i,1,g) = ...
                norm(Ao(:,:,k)-Aos_lvgl_mrf(:,:,k,i,g),'fro')^2/norm_A;
            err(k,i,2,g) = ...
                norm(Ao(:,:,k)-Aos_pgl_mrf(:,:,k,i,g),'fro')^2/norm_A;
            err(k,i,3,g) = ...
                norm(Ao(:,:,k)-Aos_lvgl_poly(:,:,k,i,g),'fro')^2/norm_A;
            err(k,i,4,g) = ...
                norm(Ao(:,:,k)-Aos_pgl_poly(:,:,k,i,g),'fro')^2/norm_A;
        end
    end
end


% figure();plot(squeeze(mean(err(:,1,2,:),1)))



mean_err = squeeze(mean(mean(err,1),4));
med_err = squeeze(median(mean(err,1),4));


% Mean error
figure();
% semilogx(Ms,mean_err(1),'-x');hold on
semilogx(Ms,mean_err(2),'-o');hold on
% semilogx(Ms,mean_err(3),'--x');hold on
% semilogx(Ms,mean_err(4),'--o');hold off
xlabel('Number of samples')
ylabel('Mean error')
legend(leg)
grid on; axis tight

% % Mean error
% figure();
% semilogx(Ms,mean_err(:,1),'-x');hold on
% semilogx(Ms,mean_err(:,2),'-o');hold on
% semilogx(Ms,mean_err(:,3),'--x');hold on
% semilogx(Ms,mean_err(:,4),'--o');hold off
% xlabel('Number of samples')
% ylabel('Mean error')
% legend(leg)
% grid on; axis tight
% 
% % Median error
% figure();
% semilogx(Ms,med_err(:,1),'-x');hold on
% semilogx(Ms,med_err(:,2),'-o');hold on
% semilogx(Ms,med_err(:,3),'--x');hold on
% semilogx(Ms,med_err(:,4),'--o');hold off
% xlabel('Number of samples')
% ylabel('Median error')
% legend(leg)
% grid on; axis tight

%% Plot fscore and recovered graphs
% fsc_glp = zeros(K,length(Ms),signal_trials);
% for g=1:signal_trials
%     for i=1:length(Ms)
%         for k=1:K
%             Ao_th = Aos(:,:,k,g);
%             Ao_th(Ao_th >= th) = 1;
%             Ao_th(Ao_th < th) = 0;
%             
%             A_aux = Aos_pgl_poly(:,:,k,i,g)./max(max(Aos_pgl_poly(:,:,k,i,g)));
%             Ao_pgl_th = A_aux;
%             Ao_pgl_th(Ao_pgl_th >= th) = 1;
%             Ao_pgl_th(Ao_pgl_th < th) = 0;
% 
%             [~,~,fsc_glp(k,i,g),~,~] = ...
%                 graph_learning_perf_eval(Ao_th,Ao_pgl_th);
%         end
%     end
% end
% 
% mean_fsc_pgl = squeeze(mean(mean(fsc_glp,1),3));
% med_fsc_pgl = squeeze(median(mean(fsc_glp,1),3));
% 
% rec_pgl = squeeze(mean(sum(fsc_glp == 1,3)/signal_trials,1));
% 
% % Mean fsc
% figure();
% semilogx(Ms,mean_fsc_pgl,'-o');
% xlabel('Number of samples')
% ylabel('Mean fscore')
% legend(leg)
% grid on; axis tight
% 
% % Median fsc
% figure();
% semilogx(Ms,med_fsc_pgl,'-o')
% xlabel('Number of samples')
% ylabel('Median fscore')
% legend(leg)
% grid on; axis tight
% 
% % Recovered graphs
% figure();
% semilogx(Ms,rec_pgl,'-o')
% xlabel('Number of samples')
% ylabel('Recovered graphs (fsc)')
% legend(leg)
% grid on; axis tight