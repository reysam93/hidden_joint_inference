%%
%rng(5)
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

prms.epsilon = 1e-6;
% Regs for reweighted
prms.delta1 = 1e-4;
prms.delta2 = 1e-4;
prms.max_iters = 5;

hid_nodes = 'min';
nG = 100;
%models = {'baseline','lowrank','lowrank rw','grouplasso','grouplasso rw'};
models = {'lowrank rw'};

% Create graphs
As = zeros(N,N,K);
err_joint = zeros(nG,numel(models),HH,K);
err_sep = zeros(nG,numel(models),HH,K);
tic
parfor g = 1:nG
    A = generate_connected_ER(N,p);
    A_org = A;
    err_joint_g = zeros(numel(models),HH,K);
    err_sep_g = zeros(numel(models),HH,K);
    
    for m = 1:numel(models)
        model = models{m};
        regs = get_regs(model,prms);
        for hd = 1:HH
            As = gen_similar_graphs(A,K,pert_links);

            Cs = create_cov(As,L,M,sampled);
        
            % Same observed/hidden nodes for all graphs
            [n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
            Ao = As(n_o,n_o,:);
            Co = Cs(n_o,n_o,:);
            
            %Estimate graph structure
            [Ao_hat,Ao_hat_sep] = estimate_A(model,Co,regs);
            
            Ao_hat = Ao_hat./max(max(Ao_hat));
            Ao_hat_sep = Ao_hat_sep./max(max(Ao_hat_sep));
            for k=1:K
                norm_Ao = norm(Ao(:,:,k),'fro')^2;
                err_joint_g(m,hd,k) = norm(Ao(:,:,k)-Ao_hat(:,:,k),'fro')^2/norm_Ao;
                err_sep_g(m,hd,k) = norm(Ao(:,:,k)-Ao_hat_sep(:,:,k),'fro')^2/norm_Ao;
            end
        end
    end
    err_joint(g,:,:,:) = err_joint_g;
    err_sep(g,:,:,:) = err_sep_g;
    
end
toc
%%
fmts = {'s-','x-','o-','*-','s:','x:','o:','*:'};
figure()
if HH == 1
    plot(squeeze(mean(err_sep,3)))
    hold on
    plot(squeeze(mean(err_joint,3)))
    legend(['Mean Error separate', num2str(mean(mean(err_sep)))],['Mean Error joint:',num2str(mean(mean(err_joint)))])
else
    res_err_sep = squeeze(median(median(err_sep,4)));
    res_err_joint = squeeze(median(median(err_joint,4)));
    for t = 1:4
        plot(1:3,res_err_joint(t+1,:),fmts{t},'MarkerSize',12,'LineWidth',2)
        hold on
        lgnd{t} = ['Joint ' models{t+1}];
    end
    for t = 1:4
        plot(1:3,res_err_sep(t+1,:),fmts{t+4},'MarkerSize',12,'LineWidth',2)
        hold on
        lgnd{t+4} = ['Sep ' models{t+1}];
    end
    legend(lgnd)
    
end
ylabel('Frobenius norm')
xlabel('Number of graphs')
title('estA-Lowrank rw')

disp([num2str(mean(mean(err_joint))), '----', num2str(mean(mean(err_sep)))])

%%

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