%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

n_graphs = 20;
sig_trials = 5;

K = 3;
N = 20;
O = 19;
p = 0.2;
F = 4;
pert_links = 3;
Ms = [1e3 1e4 1e5];
hid_nodes = 'min';
th = 0.3;

leg = {'G1, joint','G2, joint','G3, joint','G1, sep','G2, sep','G3, sep'};

max_iters = 10;
regs = struct();
regs.alpha   = 1;       % Sparsity of S
regs.gamma   = 110;      % Group Lasso
regs.beta    = 10;      % Similarity of S
regs.eta     = 10;      % Similarity of P
regs.mu      = 1000;    % Commutative penalty
regs.delta1  = 1e-3;    % Small number for reweighted

max_M = Ms(end);

tic
Aos = zeros(O,O,K,n_graphs);
Aos_joint = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
Aos_sep = zeros(O,O,K,length(Ms),sig_trials,n_graphs);
for g=1:n_graphs
    disp(['G: ' num2str(g)])
    
    % Create graphs and get hidden nodes
    A = generate_connected_ER(N,p);
    As = gen_similar_graphs(A,K,pert_links);
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
    Ao = As(n_o,n_o,:);
    Aoh = As(n_o,n_h,:);
    
    % Generate graph filters
    H = zeros(N,N,K);
    for k=1:K
        h = rand(F,1)*2-1;
        for f=1:F
            H(:,:,k) = H(:,:,k) + h(f)*As(:,:,k)^(f-1);
        end
    end
    
    Aos_joint_g = zeros(O,O,K,length(Ms),sig_trials);
    Aos_sep_g = zeros(O,O,K,length(Ms),sig_trials);
    for j=1:sig_trials
        % Generate signals X
        X = zeros(N,max_M,K);
        for k=1:K
            X(:,:,k) = H(:,:,k)*randn(N,max_M);
        end
        
        for i=1:length(Ms)
            M = Ms(i);
            disp(['   M: ' num2str(M)])
            
            % Compute covariance
            Cs = zeros(N,N,K);
            for k=1:K
                Cs(:,:,k) = X(:,1:M,k)*X(:,1:M,k)'/M;
                disp(['- k: ' num2str(k) ' norm Ck '...
                    num2str(norm(Cs(:,:,k),'fro'))])
            end
            Co = Cs(n_o,n_o,:);
            
            % Joint inference
            Ao_hat_j = estA_pgl_colsp_rw2(Co,N-O,regs,max_iters);
            Aos_joint_g(:,:,:,i,j) = Ao_hat_j./max(max(Ao_hat_j));
            % Separate inference
            for k=1:K
                Ao_hat_s = estA_pgl_colsp_rw2(Co(:,:,k),N-O,regs,max_iters);
                Aos_sep_g(:,:,k,i,j) = Ao_hat_s./max(max(Ao_hat_s));
            end
        end
    end
    Aos(:,:,:,g) = Ao;
    Aos_joint(:,:,:,:,j,g) = Aos_joint_g;
    Aos_sep(:,:,:,:,j,g) = Aos_sep_g;
end
t = toc;
disp(['----- ' num2str(t/3600) ' hours -----'])

%% Plot mean/median error
err_joint = zeros(K,length(Ms),sig_trials,n_graphs);
err_sep = zeros(K,length(Ms),sig_trials,n_graphs);
for g=1:n_graphs
    for j=1:sig_trials
        for k=1:K
            norm_A = norm(Aos(:,:,k,g),'fro')^2;
            for i=1:length(Ms)
                % Matrices normalized to maximum value 1
                err_joint(k,i,j,g) = ...
                    norm(Aos(:,:,k,g)-Aos_joint(:,:,k,i,j,g),'fro')^2/norm_A;
                err_sep(k,i,g) = ...
                    norm(Aos(:,:,k,g)-Aos_sep(:,:,k,i,j,g),'fro')^2/norm_A;
                
                %             % Try error normalizing so first column adds to 1
                %             edges_col1 = sum(Aos(:,1,k,g));
                %             if edges_col1==0
                %                 disp(['0 links in column 1! ' num2str(g) ' ' num2str(k)])
                %                 edges_col1 = 1;
                %             end
                %             Ao_norm = Aos(:,:,k,g)/edges_col1;
                %             norm_A = norm(Ao_norm,'fro')^2;
                %             Ao_j_norm = Aos_joint(:,:,k,i,g)/sum(Aos_joint(:,1,k,i,g));
                %             Ao_s_norm = Aos_sep(:,:,k,i,g)/sum(Aos_sep(:,1,k,i,g));
                %
                %             err_joint(k,i,g) = ...
                %                 norm(Ao_norm-Ao_j_norm,'fro')^2/norm_A;
                %             err_sep(k,i,g) = ...
                %                 norm(Ao_norm-Ao_s_norm,'fro')^2/norm_A;
            end
        end
    end
end

mean_err_joint = squeeze(mean(mean(err_joint,3),4));
mean_err_sep = squeeze(mean(mean(err_sep,3),4));
med_err_joint = squeeze(median(median(err_joint,3),4));
med_err_sep = squeeze(median(median(err_sep,3),4));

% Mean error
figure();
semilogx(Ms,mean_err_joint(1,:),'-o'); hold on
semilogx(Ms,mean_err_joint(2,:),'-x'); hold on
semilogx(Ms,mean_err_joint(3,:),'-v'); hold on
semilogx(Ms,mean_err_sep(1,:),'--o'); hold on
semilogx(Ms,mean_err_sep(2,:),'--x'); hold on
semilogx(Ms,mean_err_sep(3,:),'--v'); hold off
xlabel('Number of samples')
ylabel('Mean error')
legend(leg)
grid on; axis tight

% Median error
figure();
semilogx(Ms,med_err_joint(1,:),'-o'); hold on
semilogx(Ms,med_err_joint(2,:),'-x'); hold on
semilogx(Ms,med_err_joint(3,:),'-v'); hold on
semilogx(Ms,med_err_sep(1,:),'--o'); hold on
semilogx(Ms,med_err_sep(2,:),'--x'); hold on
semilogx(Ms,med_err_sep(3,:),'--v'); hold off
xlabel('Number of samples')
ylabel('Median error')
legend(leg)
grid on; axis tight

%% Plot fscore and recovered graphs
fsc_joint = zeros(K,length(Ms),sig_trials,n_graphs);
fsc_sep = zeros(K,length(Ms),sig_trials,n_graphs);
for g=1:n_graphs
    Ao_th = Aos(:,:,k,g);
    Ao_th(Ao_th >= th) = 1;
    Ao_th(Ao_th < th) = 0;
    
    for j=1:sig_trials
        for i=1:length(Ms)
            for k=1:K            
                Ao_j_th = Aos_joint(:,:,k,i,j,g);
                Ao_j_th(Ao_j_th >= th) = 1;
                Ao_j_th(Ao_j_th < th) = 0;
                Ao_s_th = Aos_sep(:,:,k,i,j,g);
                Ao_s_th(Ao_s_th >= th) = 1;
                Ao_s_th(Ao_s_th < th) = 0;
                
                [~,~,fsc_joint(k,i,j,g),~,~] = ...
                    graph_learning_perf_eval(Ao_th,Ao_j_th);
                [~,~,fsc_sep(k,i,j,g),~,~] = ...
                    graph_learning_perf_eval(Ao_th,Ao_s_th);
            end
        end
    end
end

mean_fsc_joint = squeeze(mean(mean(fsc_joint,3),4));
mean_fsc_sep = squeeze(mean(mean(fsc_sep,3),4));
med_fsc_joint = squeeze(median(median(fsc_joint,3),4));
med_fsc_sep = squeeze(median(median(fsc_joint,3),4));

rec_joint = sum(sum(fsc_joint == 1,3),4)/(n_graphs*sig_trials);
rec_sep= sum(sum(fsc_sep == 1,3),4)/(n_graphs*sig_trials);

% Mean fsc
figure();
semilogx(Ms,mean_fsc_joint(1,:),'-o'); hold on
semilogx(Ms,mean_fsc_joint(2,:),'-x'); hold on
semilogx(Ms,mean_fsc_joint(3,:),'-v'); hold on
semilogx(Ms,mean_fsc_sep(1,:),'--o'); hold on
semilogx(Ms,mean_fsc_sep(2,:),'--x'); hold on
semilogx(Ms,mean_fsc_sep(3,:),'--v'); hold off
xlabel('Number of samples')
ylabel('Mean fscore')
legend(leg)
grid on; axis tight

% Median fsc
figure();
semilogx(Ms,med_fsc_joint(1,:),'-o'); hold on
semilogx(Ms,med_fsc_joint(2,:),'-x'); hold on
semilogx(Ms,med_fsc_joint(3,:),'-v'); hold on
semilogx(Ms,med_fsc_sep(1,:),'--o'); hold on
semilogx(Ms,med_fsc_sep(2,:),'--x'); hold on
semilogx(Ms,med_fsc_sep(3,:),'--v'); hold off
xlabel('Number of samples')
ylabel('Median fscore')
legend(leg)
grid on; axis tight

% Recovered graphs
figure();
semilogx(Ms,rec_joint(1,:),'-o'); hold on
semilogx(Ms,rec_joint(2,:),'-x'); hold on
semilogx(Ms,rec_joint(3,:),'-v'); hold on
semilogx(Ms,rec_sep(1,:),'--o'); hold on
semilogx(Ms,rec_sep(2,:),'--x'); hold on
semilogx(Ms,rec_sep(3,:),'--v'); hold off
xlabel('Number of samples')
ylabel('Recovered graphs')
legend(leg)
grid on; axis tight