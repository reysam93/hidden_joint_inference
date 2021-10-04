%%
rng(1)
addpath(genpath('utils'));
addpath(genpath('opt'));

close all
Ks = [3,6];
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
prms.max_iters = 10;

hid_nodes = 'min';
nG = 64;
models = {'baseline rw','PNN rw','grouplasso rw','lowrank rw'};

% Create graphs
As = zeros(N,N,Ks(end));
err_joint = zeros(nG,numel(Ks),numel(models),HH,Ks(end));
err_sep = zeros(nG,numel(Ks),numel(models),HH,Ks(end));
tic
A_T = cell(nG,numel(Ks),numel(models),HH);
parfor g = 1:nG
    disp(['Graph: ' num2str(g)])
    A = generate_connected_ER(N,p);
    A_org = A;
    err_joint_g = zeros(numel(Ks),numel(models),HH,Ks(end));
    err_sep_g = zeros(numel(Ks),numel(models),HH,Ks(end));
    A_T_g = cell(numel(Ks),numel(models),HH);
    
    As = gen_similar_graphs(A,Ks(end),pert_links);
    Cs = create_cov(As,L,M,sampled);
    A_H = cell(numel(Ks),HH);
    for m = 1:numel(models)
        model = models{m};
        for hd = 1:HH
            regs = get_regs(model,prms,hd);
            % Same observed/hidden nodes for all graphs
            [n_o, n_h] = select_hidden_nodes(hid_nodes, N-hd, As(:,:,1));
            for nk = 1:numel(Ks)
                K = Ks(nk);
                Ao = As(n_o,n_o,1:K);
                Co = Cs(n_o,n_o,1:K);

                %Estimate graph structure
                [Ao_hat,Ao_hat_sep] = estimate_A(model,Co,regs);

                Ao_hat = Ao_hat./max(max(Ao_hat));
                Ao_hat_sep = Ao_hat_sep./max(max(Ao_hat_sep));
                A_aux = zeros(3,N-hd,N-hd,K);
                A_aux(1,:,:,:) = Ao;A_aux(2,:,:,:) = Ao_hat; A_aux(3,:,:,:) = Ao_hat_sep;  
                A_H{nk,hd} = A_aux;

                for k=1:K
                    norm_Ao = norm(Ao(:,:,k),'fro')^2;
                    %guardar los errores de K=3 y K = 6
                    err_joint_g(nk,m,hd,k) = norm(Ao(:,:,k)-Ao_hat(:,:,k),'fro')^2/norm_Ao;
                    err_sep_g(nk,m,hd,k) = norm(Ao(:,:,k)-Ao_hat_sep(:,:,k),'fro')^2/norm_Ao;
                end
            end
        end
        A_T_g(:,m,:) = A_H;
    end
    A_T(g,:,:,:) = A_T_g;
    err_joint(g,:,:,:,:) = err_joint_g;
    err_sep(g,:,:,:,:) = err_sep_g;
    
end
toc
%%
plot_exp1