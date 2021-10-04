%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

n_graphs = 20;
K = 3;
N = 20;
O = 19;
p = 0.2;
pert_links = 3;
L = 3;
M = 1e3;
sampled = false;

max_iters = 15;
regs = struct();

% regs.alpha = 1;     % sparse A
% regs.lambda = 1;    % sparse Aoh
% regs.delta1 = 1e-4; % sparse A
% regs.delta2 = 1e-4; % sparse Aoh
% regs.beta = 50;     % Ao similarity 
% regs.eta = 50;      % Aoh similarity
% regs.mu = 1000;      % Commutativity

% For H=1, C perfect --> sale demasiado sparse las oculas, 10^-4
regs.alpha = 1;     % sparse A
regs.lambda = 0.01; % sparse Aoh
regs.delta1 = 1e-4; % sparse A
regs.delta2 = 1e-4; % sparse Aoh
regs.beta = 10;     % Ao similarity 
regs.eta = 10;      % Aoh similarity
regs.mu = 1e4;      % Commutativity

hid_nodes = 'min';

Aso_true = zeros(O,O,K,n_graphs);
Asoh_true = zeros(O,N-O,K,n_graphs);
Aso_hat = zeros(O,O,K,n_graphs);
Asoh_hat = zeros(O,N-O,K,n_graphs);

norms_Ao = zeros(K,n_graphs);
norms_Co = zeros(K,n_graphs);
err_joint = zeros(K,n_graphs);
err_sep = zeros(K,n_graphs);
tic
for g=1:n_graphs
    % Create graphs
    A = generate_connected_ER(N,p);
    As = gen_similar_graphs(A,K,pert_links);
    
    % Same observed/hidden nodes for all graphs
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
    
%     links_column1 = sum(As(n_o,n_o(1),:));
%     As = As./links_column1;
    
    % Create covariances
    Cs = zeros(N,N,K);
    for k=1:K
        h = rand(L,1)*2-1;
        h = h/norm(h,1);
        H = zeros(N);
        for l=1:L
            H = H + h(l)*As(:,:,k)^(l-1);
        end
        
        if sampled
            X = H*randn(N,M);
            Cs(:,:,k) = X*X'/M;
        else
            Cs(:,:,k) = H^2;
        end
        Cs(:,:,k) = Cs(:,:,k)/norm(Cs(:,:,k),'fro');
    end
    
    Ao = As(n_o,n_o,:);
    Aoh = As(n_o,n_h,:);
    Co = Cs(n_o,n_o,:);
    Coh = Cs(n_o,n_h,:);
    
    for k=1:K
        norms_Ao(k,g) = norm(Ao(:,:,k),'fro')^2;
        norms_Co(k,g) = norm(Co(:,:,k),'fro')^2;
    end
        
    comm = norm(vec(pagemtimes(Cs,As)-pagemtimes(As,Cs)),2)^2;
    comm_obs = norm(vec(pagemtimes(Co,Ao)-pagemtimes(Ao,Co)),2)^2;
    disp(['g: ' num2str(g) '  norm(CA-AC) = ' num2str(comm)...
        '  -  norm(CoAo-AoCo) = ' num2str(comm_obs)])
        
    [Ao_hat,Aoh_hat,Coh_hat] = estA_alt_rw(Co,N-O,regs,max_iters,true);
    
    Ao_hat_sep = zeros(O,O,K);
    for k=1:K
        [Ao_hat_sep(:,:,k),~,~] = estA_alt_rw(Co(:,:,k),N-O,regs);
    end
    
    % Set maximum value to 1
    Ao_hat = Ao_hat./max(max(Ao_hat));
    Ao_hat_sep = Ao_hat_sep./max(max(Ao_hat_sep));
    
    for k=1:K
        norm_Ao = norm(Ao(:,:,k),'fro')^2;
        
        err_joint(k,g) = norm(Ao(:,:,k)-Ao_hat(:,:,k),'fro')^2/norm_Ao;
        err_sep(k,g) = norm(Ao(:,:,k)-Ao_hat_sep(:,:,k),'fro')^2/norm_Ao;
        
        if g == 1 || g==5
            figure();
            subplot(2,2,1);imagesc(Ao(:,:,k));colorbar();
            subplot(2,2,2);imagesc(Aoh(:,:,k));colorbar();
            subplot(2,2,3);imagesc(Ao_hat(:,:,k));colorbar();
            subplot(2,2,4);imagesc(Aoh_hat(:,:,k));colorbar();
            
%             figure();imagesc(Ao_hat_sep(:,:,k));colorbar();title('Separate')
        end 
    end
    disp(['   Err: ' num2str(err_joint(:,g)')])
    
    Aso_true(:,:,:,g) = Ao;
    Asoh_true(:,:,:,g) = Aoh;
    Aso_hat(:,:,:,g) = Ao_hat;
    Asoh_hat(:,:,:,g) = Aoh_hat;
end
t = toc;
disp(['--- ' num2str(t/60) ' minutes'])

%% Print summary
mean_err_joint = mean(err_joint,2);
mean_err_sep = mean(err_sep,2);

figure();
subplot(1,2,1);plot(mean(norms_Ao));subplot(1,2,2);plot(mean(norms_Co))

figure()
hold on
plot(err_joint(1,:))
plot(err_joint(2,:))
plot(err_joint(3,:))

disp(['Joint err: ' num2str(mean_err_joint')])
disp(['Separ err: ' num2str(mean_err_sep')])
disp(['Mean joint err: ' num2str(mean(mean_err_joint))])
disp(['Mean separ err: ' num2str(mean(mean_err_sep))])

median_err_joint = median(err_joint,2);
median_err_sep = median(err_sep,2);
disp(['Joint err: ' num2str(median_err_joint')])
disp(['Separ err: ' num2str(mean_err_sep')])
disp(['Median joint err: ' num2str(mean(median_err_joint))])
disp(['Median separ err: ' num2str(mean(median_err_sep))])


