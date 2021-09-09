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


regs = struct();
regs.alpha = 1;
regs.beta = 1;
regs.eta = 1;
regs.lambda = 1;
regs.mu = 1;
% regs for initialization
regs.gamma = 1;
regs.epsilon = 1e-6;

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
    Cs(:,:,k) = H^2;
end

% Same observed/hidden nodes for all graphs
[n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
Ao = As(n_o,n_o,:);
Co = Cs(n_o,n_o,:);
Aoh = As(n_o,n_h,:);
Coh = Cs(n_o,n_h,:);

tic
err = zeros(K);
err_init = zeros(K);
[Ao_hat,Aoh_hat,Coh_hat,A_init] = estA_alt(Co,N-O,regs,true);
toc

for k=1:K
    err(k) = norm(Ao(:,:,k)-Ao_hat(:,:,k),'fro')^2/norm(Ao(:,:,k),'fro')^2;
    err_init(k) = norm(Ao(:,:,k)-A_init(:,:,k),'fro')^2/norm(Ao(:,:,k),'fro')^2;
end

err

err_init