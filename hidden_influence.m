%%
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

close all
Ks = [2,6];%[3,6];   %--------------------
N = 20;
O = 15;%15;        %-------------------
HH = N-O;
p = 0.2;
pert_links = 3;
L = 3;
M = 1e4;
sampled = true;

regs = struct();
regs.lambda = 1;
% regs for initialization

prms.epsilon = 1e-6;
% Regs for reweighted
prms.delta1 = 1e-3;
prms.delta2 = 1e-3;
prms.max_iters = 10;


hid_nodes = 'min';
nG = 100;             %-----------------
%models = {'baseline','lowrank','lowrank rw','grouplasso','grouplasso rw'};
models = {'No hidden rw','PGL rw','PNN rw'};
%models = {'PNN rw'};

% Create graphs
As = zeros(N,N,Ks(end));
tic
parfor g = 1:nG
    disp(['Graph: ' num2str(g)])
    A = generate_connected_ER(N,p);
    err_joint_K = {};
    %err_joint_gk2 = zeros(numel(models),HH,Ks(2));
    %err_joint_gk3 = zeros(numel(models),HH,Ks(3));
    
    As = gen_similar_graphs(A,Ks(end),pert_links);
    
    %%%%%%%% create covariance matrix
    K = Ks(end);
    Cs = zeros(size(As));
    %Cs = create_cov(As,L,M,sampled);
    hs = zeros(L,K);
    for k=1:K
        hs(:,k) = randn(L,1);
        hs(:,k) = hs(:,k)/norm(hs(:,k),1);
    end
    for k=1:K
        H = zeros(N);
        for l=1:L
            H = H + hs(l,k)*As(:,:,k)^(l-1);
        end
        C_true = H^2;

        X = sqrtm(C_true)*randn(N,M);
        Cs(:,:,k) = X*X'/(M-1);
    end
    %%%%%%%%%%%%%%%
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
                [Ao_hat,~] = estimate_A(model,Co,regs);
                Ao_hat = Ao_hat./max(max(Ao_hat));
                for k=1:K
                    norm_Ao = norm(Ao(:,:,k),'fro')^2;
                    err_joint_K{nk}(m,hd,k) = norm(Ao(:,:,k)-Ao_hat(:,:,k),'fro')^2/norm_Ao;
                end
            end           
        end
    end
    err_joint{g} = err_joint_K;
end
toc
%%
for g = 1:nG
    for k = 1:numel(Ks)
        error_joint(g,k,:,:) = mean(cell2mat(err_joint{g}(k)),3);
    end
end
all_errors = squeeze(mean(error_joint));
%%
figure()
for k = 1:numel(Ks)
    plot(squeeze(all_errors(k,:,:))','linewidth',2)
    hold on
end
legend('No-H K=2', 'PGL K=2', 'PNN K=2','No-H K=6', 'PGL K=6', 'PNN K=6')
grid on
%%
save('data_exp1.mat');
%%
load('data_exp1.mat');
%plot_exp1