%%
rng(5)
addpath(genpath('utils'));
addpath(genpath('opt'));

K = 3;
N = 20;
O = 19;
HH = N-O;
p = 0.2;
pert_links = 3;
L = 3;
M = 1e3;
sampled = false;

regs = struct();
regs.lambda = 1;
% regs for initialization

regs.epsilon = 1e-6;
% Regs for reweighted
regs.delta1 = 1e-3;
regs.delta2 = 1e-3;
regs.max_iters = 10;

hid_nodes = 'min';
nG = 10;
model = 'lowrank rw';

if strcmp(model,'lowrank')
    regs.alpha = 1e-3;
    regs.gamma = 10; 
    regs.beta = 1;  
    regs.eta = 1e-1;
elseif strcmp(model,'lowrank rw')
    regs.alpha = 1e-2;
    regs.gamma = 1; 
    regs.beta = 1e-1;  
    regs.eta = 1e-1;
    regs.mu = 1e2; 
end
% Create graphs
As = zeros(N,N,K);
err_joint = zeros(nG,HH,K);
err_sep = zeros(nG,HH,K);
tic
for g = 1:nG
    As(:,:,1) = generate_connected_ER(N,p);
    A_org = As(:,:,1);
    for hd = 1:HH
        for k=2:K
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
%         figure()
%         for k = 1:K
%             subplot(2,K,k)
%             imagesc(As(:,:,k))
%             title(num2str(k))
%             subplot(2,K,k+K)
%             imagesc(As(:,:,k)-A_org)
%             title(num2str(k))
%         end
        % Plot graphs
        % figure();
        % for k=1:K
        %     subplot(1,K,k);imagesc(As(:,:,k));colorbar();title(['A' num2str(k)])
        % end

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

        % comm = norm(vec(pagemtimes(Cs,As)-pagemtimes(As,Cs)),'fro')^2;
        % comm_obs = norm(vec(pagemtimes(Co,Ao)-pagemtimes(Ao,Co)),'fro')^2;
        % disp(['norm(CA-AC) = ' num2str(comm) '  -  norm(CoAo-AoCo) = '...
        %     num2str(comm_obs)])

        Ao_hat = zeros(O,O,K);
        Ao_hat_sep = zeros(O,O,K);
        %tic
        if strcmp(model,'lowrank')
            [Ao_hat,~] = estA_lowrank(Co,regs);
            for k=1:K
                [Ao_hat_sep(:,:,k),~] = estA_lowrank(Co(:,:,k),regs);
            end
        elseif strcmp(model,'lowrank rw')
            [Ao_hat,~] = estA_lowrank_rw(Co,regs);
            for k=1:K
                [Ao_hat_sep(:,:,k),~] = estA_lowrank_rw(Co(:,:,k),regs);
            end
        end
        %toc

        Ao_hat = Ao_hat./max(max(Ao_hat));
        Ao_hat_sep = Ao_hat_sep./max(max(Ao_hat_sep));
        for k=1:K
            norm_Ao = norm(Ao(:,:,k),'fro')^2;
            err_joint(g,hd,k) = norm(Ao(:,:,k)-Ao_hat(:,:,k),'fro')^2/norm_Ao;
            err_sep(g,hd,k) = norm(Ao(:,:,k)-Ao_hat_sep(:,:,k),'fro')^2/norm_Ao;
        end
    end
end
toc

figure()
if HH == 1
    plot(squeeze(mean(err_sep,3)))
    hold on
    plot(squeeze(mean(err_joint,3)))
    legend(['Mean Error separate', num2str(mean(mean(err_sep)))],['Mean Error joint:',num2str(mean(mean(err_joint)))])
else
    plot(mean(mean(err_sep,3)))
    hold on
    plot(mean(mean(err_joint,3)))
    legend('Mean Error separate','Mean Error joint')
    
end
ylabel('Frobenius norm')
xlabel('Number of graphs')
title('estA-Lowrank')

disp([num2str(mean(mean(err_joint))), '----', num2str(mean(mean(err_sep)))])

figure()
for k = 1:K
    subplot(3,3,k)
    imagesc(Ao(:,:,k))
    subplot(3,3,k+3)
    imagesc(Ao_hat(:,:,k))
    subplot(3,3,k+6)
    imagesc(Ao_hat_sep(:,:,k))
end
%%

err_init = zeros(K,1);
% Set maximum value to 1


for k=1:K
    norm_Ao = norm(Ao(:,:,k),'fro')^2;
    err_joint(k) = norm(Ao(:,:,k)-Ao_hat(:,:,k),'fro')^2/norm_Ao;
    %err_init(k) = norm(Ao(:,:,k)-A_init(:,:,k),'fro')^2/norm_Ao;
    err_sep(k) = norm(Ao(:,:,k)-Ao_hat_sep(:,:,k),'fro')^2/norm_Ao;
    figure();
    subplot(1,2,1);imagesc(Ao(:,:,k));colorbar();
    %subplot(2,2,2);imagesc(Aoh(:,:,k));colorbar();
    subplot(1,2,2);imagesc(Ao_hat(:,:,k));colorbar();
    %subplot(2,2,4);imagesc(Aoh_hat(:,:,k));colorbar();
    
    figure();imagesc(Ao_hat_sep(:,:,k));colorbar();title('Separate')
end

disp(['Joint err: ' num2str(err_joint')])
disp(['Separ err: ' num2str(err_sep')])
disp(['Mean joint err: ' num2str(mean(err_joint))])
disp(['Mean separ err: ' num2str(mean(err_sep))])