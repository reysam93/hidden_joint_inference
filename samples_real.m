%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

graphs = [7 9 12];
sig_trials = 1;
K = length(graphs);
N = 32;
O = 31;
F = 4;
Ms = [1e3 1e4] %round(logspace(2,6,9));
max_M = Ms(end);
hid_nodes = 'min';
max_iters = 10;

leg = {'G1 joint','G2 joint','G3 joint',...
    'G1 sep','G2 sep','G3 sep'};

regs = struct();
regs.alpha   = 1;       % Sparsity of S
regs.gamma   = 100;      % Group Lasso
regs.beta    = 50;      % Similarity of S
regs.eta     = 25;      % Similarity of P
regs.mu      = 1000;    % Commutative penalty
regs.delta1  = 1e-3;    % Small number for reweighted

% Load graphs and generate graph filters
As = get_student_networks_graphs(graphs,N);
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
    disp(['Trial: ' num2str(j)])
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
        Ao_hat_j = estA_pgl_colsp_rw(Co,regs,max_iters);
        Aos_joint_t(:,:,:,i) = Ao_hat_j./max(max(Ao_hat_j));
        
        % Separate inference
        for k=1:K
            Ao_hat_s = estA_pgl_colsp_rw(Co(:,:,k),regs,max_iters);
            Aos_sep_t(:,:,k,i) = Ao_hat_s./max(max(Ao_hat_s));
        end
    end
    Aos_joint(:,:,:,:,j) = Aos_joint_t;
    Aos_sep(:,:,:,:,j) = Aos_sep_t;
end

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

mean_errj = mean(err_joint,3);
mean_errs = mean(err_sep,3);

% Plot properties
mark_s = 8;
line_w = 2;

% Mean error
set(0,'defaultTextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
figure();
semilogx(Ms,mean_errj(1,:),'-o','LineWidth',line_w,'MarkerSize',mark_s);hold on
semilogx(Ms,mean_errj(2,:),'-x','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_errj(3,:),'-v','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_errs(1,:),':o','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_errs(2,:),':x','LineWidth',line_w,'MarkerSize',mark_s); hold on
semilogx(Ms,mean_errs(3,:),':v','LineWidth',line_w,'MarkerSize',mark_s); hold off
xlabel('(c) Number of samples')
ylabel('Mean error')
legend(leg,'Location','northeast','NumColumns',2)
grid on;
ylim([0 1])
set(gca,'FontSize',14);
set(gcf, 'PaperPositionMode', 'auto')
