%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

K = 3;
N = 20;
O = 19;
p = 0.2;
pert_links = 3;
L = 3;
M = 1e3;
sampled = true;

regs = struct();
regs.alpha = 1;
regs.beta = 50;  % 10
regs.eta = 50;   % 10
regs.lambda = 5;
regs.mu = 1000; %100 for true C
% regs for initialization
regs.gamma = 1;
regs.epsilon = 1e-6;
% Regs for reweighted
regs.delta1 = 1e-3;
regs.delta2 = 1e-3;

hid_nodes = 'min';

% Create graphs
As = zeros(N,N,K);
As(:,:,1) = generate_connected_ER(N,p);

for k=1:K
    % rewire links
    As(:,:,k) = As(:,:,1);
    for i=1:pert_links
        node_id = randi(N);
        
        % delete link
        [link_nodes,~] = find(As(:,node_id,1)~=0);
        del_node = link_nodes(randperm(length(link_nodes),1));
        As(node_id,del_node,k) = 0;
        As(del_node,node_id,k) = 0;
        
        % create link
        [nonlink_nodes,~] = find(As(:,node_id,1)==0);
        nonlink_nodes(nonlink_nodes==node_id) = [];
        add_node = nonlink_nodes(randperm(length(nonlink_nodes),1));
        As(node_id,add_node,k) = 1;
        As(add_node,node_id,k) = 1;
%         As(:,:,k) = As(:,:,k)/sum(As(:,1,1));
    end
    assert(sum(sum(As(:,:,k)~=As(:,:,k)'))==0,'Non-symmetric matrix')
    assert(sum(diag(As(:,:,k)))==0,'Non-zero diagonal')
end

% Plot graphs
figure();
for k=1:K
    subplot(1,K,k);imagesc(As(:,:,k));colorbar();title(['A' num2str(k)])
end

% Create covariances
Cs = zeros(N,N,K);
for k=1:K
    h = rand(L,1)*2-1;
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
end

% Same observed/hidden nodes for all graphs
[n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
Ao = As(n_o,n_o,:);
Co = Cs(n_o,n_o,:);
Aoh = As(n_o,n_h,:);
Coh = Cs(n_o,n_h,:);

comm = norm(vec(pagemtimes(Cs,As)-pagemtimes(As,Cs)),'fro')^2;
comm_obs = norm(vec(pagemtimes(Co,Ao)-pagemtimes(Ao,Co)),'fro')^2;
disp(['norm(CA-AC) = ' num2str(comm) '  -  norm(CoAo-AoCo) = '...
    num2str(comm_obs)])

tic
err_joint = zeros(K,1);
err_sep = zeros(K,1);
err_init = zeros(K,1);
% [Ao_hat,Aoh_hat,Coh_hat,A_init] = estA_alt(Co,N-O,regs,true);
[Ao_hat,Aoh_hat,Coh_hat,A_init] = estA_alt_rw(Co,N-O,regs,true);

Ao_hat_sep = zeros(O,O,K);
for k=1:K
    [Ao_hat_sep(:,:,k),~,~,~] = estA_alt_rw(Co(:,:,k),N-O,regs);
end
toc

% Set maximum value to 1
Ao_hat = Ao_hat./max(max(Ao_hat));
A_init = A_init./max(max(A_init));
Ao_hat_sep = Ao_hat_sep./max(max(Ao_hat_sep));

for k=1:K
    norm_Ao = norm(Ao(:,:,k),'fro')^2;
    err_joint(k) = norm(Ao(:,:,k)-Ao_hat(:,:,k),'fro')^2/norm_Ao;
    err_init(k) = norm(Ao(:,:,k)-A_init(:,:,k),'fro')^2/norm_Ao;
    err_sep(k) = norm(Ao(:,:,k)-Ao_hat_sep(:,:,k),'fro')^2/norm_Ao;
    figure();
    subplot(2,2,1);imagesc(Ao(:,:,k));colorbar();
    subplot(2,2,2);imagesc(Aoh(:,:,k));colorbar();
    subplot(2,2,3);imagesc(Ao_hat(:,:,k));colorbar();
    subplot(2,2,4);imagesc(Aoh_hat(:,:,k));colorbar();
    
    figure();imagesc(Ao_hat_sep(:,:,k));colorbar();title('Separate')
end

disp(['Joint err: ' num2str(err_joint')])
disp(['Separ err: ' num2str(err_sep')])
disp(['Mean joint err: ' num2str(mean(err_joint))])
disp(['Mean separ err: ' num2str(mean(err_sep))])
