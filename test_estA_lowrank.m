%%
rng(5)
addpath(genpath('utils'));
addpath(genpath('opt'));

close all
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
nG = 2;
model = 'grouplasso rw';

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
elseif strcmp(model,'grouplasso')
    regs.alpha = 1e-3;
    regs.gamma = 10; 
    regs.beta = 1;  
    regs.eta = 1e-1;
elseif strcmp(model,'grouplasso rw')
    regs.alpha = 1e-2;
    regs.gamma = 1; 
    regs.beta = 1e-1;  
    regs.eta = 1e-1;
    regs.mu = 1e2; 
elseif strcmp(model,'baseline')
    regs.alpha = 1; 
    regs.beta = 1;  
end
% Create graphs
As = zeros(N,N,K);
err_joint = zeros(nG,HH,K);
err_sep = zeros(nG,HH,K);
tic
for g = 1:nG
    A = generate_connected_ER(N,p);
    A_org = A;
    for hd = 1:HH
        As = gen_similar_graphs(A,K,pert_links);

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
        elseif strcmp(model,'baseline')
            [Ao_hat,~] = estA_baseline(Co,regs);
            for k=1:K
                [Ao_hat_sep(:,:,k),~] = estA_baseline(Co(:,:,k),regs);
            end
        elseif strcmp(model,'grouplasso')
            [Ao_hat,~] = estA_grouplasso(Co,regs);
            for k=1:K
                [Ao_hat_sep(:,:,k),~] = estA_grouplasso(Co(:,:,k),regs);
            end
        elseif strcmp(model,'grouplasso rw')
            [Ao_hat,~] = estA_grouplasso_rw(Co,regs);
            for k=1:K
                [Ao_hat_sep(:,:,k),~] = estA_grouplasso_rw(Co(:,:,k),regs);
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
%%
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
title('estA-Lowrank rw')

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