%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

graphs = [7 9 12];
signal_trials = 1;
N = 32;
O = 31;
Ms = [1e4];
F = 3;
K = length(graphs);
max_M = Ms(end);
hid_nodes = 'min';
th = 0.3;

leg = {'Pgl,C_{poly}'};


max_iters = 10;
regs = struct();
regs.alpha   = 1;       % Sparsity of S
regs.gamma   = 110;      % Group Lasso
regs.beta    = 10;      % Similarity of S
regs.eta     = 10;      % Similarity of P
regs.mu      = 1000;    % Commutative penalty
regs.delta1  = 1e-2;    % Small number for reweighted

As = get_student_networks_graphs(graphs,N);

% Graph filter
H = zeros(N,N,K);
for k=1:K
    h = rand(F,1)*2-1;
    for f=1:F
        H(:,:,k) = H(:,:,k) + h(f)*(As(:,:,k)^(f-1));
    end
end

% Try with whole graph also
[n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
Ao = As(n_o,n_o,:);
Aoh = As(n_o,n_h,:);

tic
Aos = zeros(O,O,K,signal_trials);
Aos_pgl = zeros(O,O,K,length(Ms),signal_trials);
for g=1:signal_trials
    disp(['Trial: ' num2str(g)])
    
    
    
    % Generate signals X
    X = zeros(N,max_M,K);
    for k=1:K
        X(:,:,k) = H(:,:,k)*randn(N,max_M);
    end
    
    Aos_pgl_g = zeros(O,O,K,length(Ms));
    for i=1:length(Ms)
        M = Ms(i);
        
        % Compute covariance
        Cs = zeros(N,N,K);
        for k=1:K
            Cs(:,:,k) = X(:,1:M,k)*X(:,1:M,k)'/M;
        end
        Co = Cs(n_o,n_o,:);
        
        % Pgl rw
        [Aos_pgl_g(:,:,:,i),P_pgl] = estA_pgl_colsp_rw2(Co,N-O,regs,max_iters);
        % Do this line when computing the error
%         Aos_joint_g(:,:,:,i) = Ao_hat_j./max(max(Ao_hat_j));
    end
    
    Aos(:,:,:,g) = Ao;
    Aos_pgl(:,:,:,:,g) = Aos_pgl_g;
end
t = toc;
disp(['----- ' num2str(t/3600) ' hours -----'])


%% Plot mean/median error
err_pgl = zeros(K,length(Ms),signal_trials);
for g=1:signal_trials
    for k=1:K
        norm_A = norm(Aos(:,:,k,g),'fro')^2;
        for i=1:length(Ms)
            % Matrices set to maximum value 1
            A_aux = Aos_pgl(:,:,k,i,g)./max(max(Aos_pgl(:,:,k,i,g)));
            err_pgl(k,i,g) = ...
                norm(Aos(:,:,k,g)-A_aux,'fro')^2/norm_A;
            
            % Try error normalizing so first column adds to 1
%             Ao_norm = Aos(:,:,k,g)/sum(Aos(:,1,k,g));
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
    if g==1 || g==2
        figure()
        for k=1:K
            subplot(2,K,k)
            imagesc(Aos_pgl(:,:,k,end,g))
            colorbar()
            title(['A: ' num2str(k)])
            subplot(2,K,K+k)
            imagesc(P_pgl(:,:,k))
            colorbar()
            title(['P: ' num2str(k)])
        end
    end
end

mean_err_pgl = squeeze(mean(mean(err_pgl,1),3));
med_err_pgl = squeeze(median(mean(err_pgl,1),3));

% Mean error
figure();
semilogx(Ms,mean_err_pgl,'-o');
xlabel('Number of samples')
ylabel('Mean error')
legend(leg)
grid on; axis tight

% Median error
figure();
semilogx(Ms,med_err_pgl,'-o')
xlabel('Number of samples')
ylabel('Median error')
legend(leg)
grid on; axis tight

%% Plot fscore and recovered graphs
fsc_glp = zeros(K,length(Ms),signal_trials);
for g=1:signal_trials
    for i=1:length(Ms)
        for k=1:K
            Ao_th = Aos(:,:,k,g);
            Ao_th(Ao_th >= th) = 1;
            Ao_th(Ao_th < th) = 0;
            
            A_aux = Aos_pgl(:,:,k,i,g)./max(max(Aos_pgl(:,:,k,i,g)));
            Ao_pgl_th = A_aux;
            Ao_pgl_th(Ao_pgl_th >= th) = 1;
            Ao_pgl_th(Ao_pgl_th < th) = 0;

            [~,~,fsc_glp(k,i,g),~,~] = ...
                graph_learning_perf_eval(Ao_th,Ao_pgl_th);
        end
    end
end

mean_fsc_pgl = squeeze(mean(mean(fsc_glp,1),3));
med_fsc_pgl = squeeze(median(mean(fsc_glp,1),3));

rec_pgl = squeeze(mean(sum(fsc_glp == 1,3)/signal_trials,1));

% Mean fsc
figure();
semilogx(Ms,mean_fsc_pgl,'-o');
xlabel('Number of samples')
ylabel('Mean fscore')
legend(leg)
grid on; axis tight

% Median fsc
figure();
semilogx(Ms,med_fsc_pgl,'-o')
xlabel('Number of samples')
ylabel('Median fscore')
legend(leg)
grid on; axis tight

% Recovered graphs
figure();
semilogx(Ms,rec_pgl,'-o')
xlabel('Number of samples')
ylabel('Recovered graphs')
legend(leg)
grid on; axis tight