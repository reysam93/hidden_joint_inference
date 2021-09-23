%%
%rng(5)
addpath(genpath('utils'));
addpath(genpath('opt'));

close all
K = 3;
N = 20;
O = 15;
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
prms.delta1 = 1e-3;
prms.delta2 = 1e-3;
prms.max_iters = 5;

hid_nodes = 'min';
nG = 12;
%models = {'baseline','lowrank','lowrank rw','grouplasso','grouplasso rw'};
models = {'lowrank','grouplasso'};

% Create graphs
As = zeros(N,N,K);
err_joint = zeros(nG,numel(models),HH,K);
err_sep = zeros(nG,numel(models),HH,K);
tic
A_T = cell(nG,numel(models),HH);
parfor g = 1:nG
    A = generate_connected_ER(N,p);
    A_org = A;
    err_joint_g = zeros(numel(models),HH,K);
    err_sep_g = zeros(numel(models),HH,K);
    A_T_g = cell(numel(models),HH);
    for m = 1:numel(models)
        model = models{m};
        regs = get_regs(model,prms);
        As = gen_similar_graphs(A,K,pert_links);
        Cs = create_cov(As,L,M,sampled);
        A_H = cell(1,5);
        for hd = 1:HH
        
            % Same observed/hidden nodes for all graphs
            [n_o, n_h] = select_hidden_nodes(hid_nodes, N-hd, As(:,:,1));
            Ao = As(n_o,n_o,:);
            Co = Cs(n_o,n_o,:);
            
            %Estimate graph structure
            [Ao_hat,Ao_hat_sep] = estimate_A(model,Co,regs);
            
            Ao_hat = Ao_hat./max(max(Ao_hat));
            Ao_hat_sep = Ao_hat_sep./max(max(Ao_hat_sep));
            A_aux = zeros(3,N-hd,N-hd,3);
            A_aux(1,:,:,:) = Ao;A_aux(2,:,:,:) = Ao_hat; A_aux(3,:,:,:) = Ao_hat_sep;  
            A_H{hd} = A_aux;
            
            for k=1:K
                norm_Ao = norm(Ao(:,:,k),'fro')^2;
                err_joint_g(m,hd,k) = norm(Ao(:,:,k)-Ao_hat(:,:,k),'fro')^2/norm_Ao;
                err_sep_g(m,hd,k) = norm(Ao(:,:,k)-Ao_hat_sep(:,:,k),'fro')^2/norm_Ao;
            end
        end
        A_T_g(m,:) = A_H;
    end
    A_T(g,:,:) = A_T_g;
    err_joint(g,:,:,:) = err_joint_g;
    err_sep(g,:,:,:) = err_sep_g;
    
end
toc

plot_results