%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

graphs = [7 9 12];
sig_trials = 30;
K = length(graphs);
N = 32;
O = 31;
F = 4;
Ms = round(logspace(2,6,9));
max_M = Ms(end);
hid_nodes = 'min';
max_iters = 10;

leg = {'G1 joint','G2 joint','G3 joint','G1 sep','G2 sep','G3 sep'};

regs = struct();
regs.alpha   = 1;       % Sparsity of S
regs.gamma   = 100;      % Group Lasso
regs.beta    = 50;      % Similarity of S
regs.eta     = 25;      % Similarity of P
regs.mu      = 1000;    % Commutative penalty
regs.delta1  = 1e-3;    % Small number for reweighted

% Load graphs and generate graph filters
As = get_student_networks_graphs(graphs,N);
% Try with whole graph also
[n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
Ao = As(n_o,n_o,:);

H = zeros(N,N,K);
for k=1:K
    h = rand(F,1)*2-1;
    for f=1:F
        H(:,:,k) = H(:,:,k) + h(f)*As(:,:,k)^(f-1);
    end
end

tic
Aos_joint = zeros(O,O,K,length(Ms),sig_trials);
Aos_sep = zeros(O,O,K,length(Ms),sig_trials);
for j=1:sig_trials
    % Generate signals X
    X = zeros(N,max_M,K);
    for k=1:K
        X(:,:,k) = H(:,:,k)*randn(N,max_M);
    end
    
    Aos_joint_t = zeros(O,O,K,length(Ms));
    Aos_sep_t = zeros(O,O,K,length(Ms));
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
        Aos_joint_t(:,:,:,i) = Ao_hat_j./max(max(Ao_hat_j));
        % Separate inference
        for k=1:K
            Ao_hat_s = estA_pgl_colsp_rw2(Co(:,:,k),N-O,regs,max_iters);
            Aos_sep_t(:,:,k,i) = Ao_hat_s./max(max(Ao_hat_s));
        end
    end
end
Aos_joint(:,:,:,:,j) = Aos_joint_t;
Aos_sep(:,:,:,:,j) = Aos_sep_t;

t = toc;
disp(['----- ' num2str(t/3600) ' hours -----'])

%% Plot mean/median error
err_joint = zeros(K,length(Ms),sig_trials);
err_sep = zeros(K,length(Ms),sig_trials);
for j=1:sig_trials
    for k=1:K
        norm_A = norm(Ao(:,:,k),'fro')^2;
        for i=1:length(Ms)
            % Matrices normalized to maximum value 1
            err_joint(k,i,j) = ...
                norm(Ao(:,:,k)-Aos_joint(:,:,k,i,j),'fro')^2/norm_A;
            err_sep(k,i,j) = ...
                norm(Ao(:,:,k)-Aos_sep(:,:,k,i,j),'fro')^2/norm_A;
        end
    end
end

mean_err_joint = mean(err_joint,3);
mean_err_sep = mean(err_sep,3);
med_err_joint = median(err_joint,3);
med_err_sep = median(err_sep,3);

rec_joint = sum(err_joint <= .1,3)/(sig_trials);
rec_sep= sum(err_sep <= .1,3)/(sig_trials);

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

% Median error
figure();
semilogx(Ms,rec_joint(1,:),'-o'); hold on
semilogx(Ms,rec_joint(2,:),'-x'); hold on
semilogx(Ms,rec_joint(3,:),'-v'); hold on
semilogx(Ms,rec_sep(1,:),'--o'); hold on
semilogx(Ms,rec_sep(2,:),'--x'); hold on
semilogx(Ms,rec_sep(3,:),'--v'); hold off
xlabel('Number of samples')
ylabel('Recovered graphs (err)')
legend(leg)
grid on; axis tight
