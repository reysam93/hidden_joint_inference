clear; clc;
addpath(['../../../Code/CVX/cvx'])
addpath(genpath('utils'));
% addpath(genpath('opt'));
rng(10);
%--------------------------------------------------------------------------

%------------------------
num_graph_trials = 5;
num_signal_trials = 5;
%------------------------

%------------------------
signal_range = round(logspace(2,5,4));
% signal_range = 1e3;
num_signal_range = length(signal_range);
%------------------------

%--------------------------------------------------------------------------
% Set parameters

%------------------------
% Graph parameters
K = 3;
N = 20;
O = 19;

S_true = cell(num_graph_trials,1);
P_true = cell(num_graph_trials,1);
So_true = cell(num_graph_trials,1);
Soh_true = cell(num_graph_trials,1);

p=.2;
pert_links=3;
hid_nodes = 'min';
%------------------------

%------------------------
% Estimation parameters
regs.alpha   = 1;
regs.beta    = 50;
regs.eta     = 50;
regs.lambda  = 5;
regs.mu      = 1000;
regs.gamma   = 50;
regs.epsilon = 1e-6;
regs.delta1  = 1e-3;
regs.delta2  = 1e-3;

%------------------------

%------------------------
% Filter parameters
F = 3;
% h = rand(F,1)*2-1;
%------------------------

%------------------------
low_tri_ind = find(tril(ones(N,N))-eye(N));
%------------------------

%------------------------
So_joi_alt = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_joi_alt = cell(num_graph_trials,num_signal_trials,num_signal_range);
So_sep_alt = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_sep_alt = cell(num_graph_trials,num_signal_trials,num_signal_range);

So_joi_alt_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_joi_alt_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);
So_sep_alt_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_sep_alt_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);

So_joi_pgl_colsp = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_joi_pgl_colsp = cell(num_graph_trials,num_signal_trials,num_signal_range);
So_sep_pgl_colsp = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_sep_pgl_colsp = cell(num_graph_trials,num_signal_trials,num_signal_range);

So_joi_pgl_colsp_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_joi_pgl_colsp_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);
So_sep_pgl_colsp_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_sep_pgl_colsp_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);

So_joi_pnn_colsp = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_joi_pnn_colsp = cell(num_graph_trials,num_signal_trials,num_signal_range);
So_sep_pnn_colsp = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_sep_pnn_colsp = cell(num_graph_trials,num_signal_trials,num_signal_range);

So_joi_pnn_colsp_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_joi_pnn_colsp_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);
So_sep_pnn_colsp_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_sep_pnn_colsp_rw = cell(num_graph_trials,num_signal_trials,num_signal_range);
%------------------------

%% ------------------------------------------------------------------------
%--------------------------------------------------------------------------
% GRAPH TRIALS:
for graph_trial_idx=1:num_graph_trials
    fprintf(['Graph trial ',num2str(graph_trial_idx),' of ',num2str(num_graph_trials),'\n'])

    S=zeros(N,N,K);
    S_true{graph_trial_idx} = zeros(N,N,K);

    S(:,:,1) = generate_connected_ER(N,p);
    for k=1:K
        % rewire links
        S(:,:,k) = S(:,:,1);
        for i=1:pert_links
            node_id = randi(N);

            % delete link
            [link_nodes,~] = find(S(:,node_id,1)~=0);
            del_node = link_nodes(randperm(length(link_nodes),1));
            S(node_id,del_node,k) = 0;
            S(del_node,node_id,k) = 0;

            % create link
            [nonlink_nodes,~] = find(S(:,node_id,1)==0);
            nonlink_nodes(nonlink_nodes==node_id) = [];
            add_node = nonlink_nodes(randperm(length(nonlink_nodes),1));
            S(node_id,add_node,k) = 1;
            S(add_node,node_id,k) = 1;
    %         As(:,:,k) = As(:,:,k)/sum(As(:,1,1));
        end
%         S(:,:,k) = rewired(S(:,:,1),pert_links);
%         S(:,:,k) = S(:,:,k)/sum(S(:,1,k));
        S_true{graph_trial_idx}(:,:,k) = S(:,:,k);
    end

    if graph_trial_idx==1
        figure();
        for k=1:K
            subplot(1,K,k);
            imagesc(S_true{graph_trial_idx}(:,:,k))
            colorbar();
            title(['Graph ',num2str(k)]);
        end
    end

    %------------------------

    %------------------------
    % Graph filters
	H = zeros(N,N,K);
    for k=1:K
        h = rand(F,1)*2-1;
        for f=1:F
            H(:,:,k) = H(:,:,k) + h(f)*(S(:,:,k)^(f-1));
        end
    end
    %------------------------

    %------------------------
    % Covariance matrices
	C = zeros(N,N,K);
    for k=1:K
        C(:,:,k) = H(:,:,k)^2;
    end
    %------------------------

    %------------------------
    % Select hidden nodes
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, S(:,:,1));
    So  = S(n_o,n_o,:);
    Soh = S(n_o,n_h,:);
    Co  = C(n_o,n_o,:);
    Coh = C(n_o,n_h,:);

    So_true{graph_trial_idx}  = S_true{graph_trial_idx}(n_o,n_o,:);
    Soh_true{graph_trial_idx} = S_true{graph_trial_idx}(n_o,n_h,:);
    %------------------------

    %------------------------
    % P matrices
    P_true{graph_trial_idx} = zeros(O,O,K);
    for k=1:K
        P_true{graph_trial_idx}(:,:,k) = Coh(:,:,k)*(Soh(:,:,k)');
    end
    %------------------------

    %------------------------
    % Commutativity check
    comm = norm(vec(pagemtimes(C,S)-pagemtimes(S,C)),'fro')^2;
    comm_obs = norm(vec(pagemtimes(Co,So)-pagemtimes(So,Co)),'fro')^2;
    disp(['norm(CA-AC) = ' num2str(comm) '  -  norm(CoAo-AoCo) = '...
        num2str(comm_obs)])

    %------------------------

    %% --------------------------------------------------------------------
    %----------------------------------------------------------------------
    % SIGNAL TRIALS:
    for signal_trial_idx=1:num_signal_trials
        fprintf(['    Signal trial ',num2str(signal_trial_idx),' of ',num2str(num_signal_trials),'\n'])

        %------------------------
        X_MAX = cell(K,1);
        for k=1:K
            X_MAX{k} = H(:,:,k)*randn(N,signal_range(end));
        end
        %------------------------

        %----------------------------------------------------------------------
        % NUMBER OF SIGNALS TRIALS:
        for num_signal_idx=1:num_signal_range
            fprintf(['        Number of signals ',num2str(signal_range(num_signal_idx)),'\n'])
            num_signals = signal_range(num_signal_idx);
            X = cell(K,1);
            for k=1:K
                X{k} = X_MAX{k}(:,1:num_signals);
            end

            %----------------------------------------------------------------------
            % ESTIMATION:

            %------------------------
            % Sample covariance
            C_est = zeros(N,N,K);
            for k=1:K
                C_est(:,:,k)=X{k}*X{k}'/num_signals;
            end
            Co_est = C_est(n_o,n_o,:);
            %------------------------

            %------------------------
            % Separate inference:
            fprintf(['        Separate inference\n'])

            fprintf(['        Alter. Min.\n'])
            So_hat_sep_alt = zeros(O,O,K);
            Soh_hat_sep_alt = zeros(O,N-O,K);
            Coh_hat_sep_alt = zeros(O,N-O,K);
            for k=1:K
                [So_hat_sep_alt(:,:,k),Soh_hat_sep_alt(:,:,k),Coh_hat_sep_alt(:,:,k)] = ...
                    estA_alt(Co_est(:,:,k),N-O,regs);
            end
            So_sep_alt{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_sep_alt;
            P_sep_alt{graph_trial_idx,signal_trial_idx,num_signal_idx} = zeros(O,O,K);
            for k=1:K
                P_sep_alt{graph_trial_idx,signal_trial_idx,num_signal_idx}(:,:,k)=Coh_hat_sep_alt(:,:,k)*Soh_hat_sep_alt(:,:,k)';
            end

            fprintf(['        Alter. Min. with RW\n'])
            So_hat_sep_alt_rw = zeros(O,O,K);
            Soh_hat_sep_alt_rw = zeros(O,N-O,K);
            Coh_hat_sep_alt_rw = zeros(O,N-O,K);
            for k=1:K
                [So_hat_sep_alt_rw(:,:,k),Soh_hat_sep_alt_rw(:,:,k),Coh_hat_sep_alt_rw(:,:,k)] = ...
                    estA_alt_rw(Co_est(:,:,k),N-O,regs);
            end
            So_sep_alt_rw{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_sep_alt_rw;
            P_sep_alt_rw{graph_trial_idx,signal_trial_idx,num_signal_idx} = zeros(O,O,K);
            for k=1:K
                P_sep_alt_rw{graph_trial_idx,signal_trial_idx,num_signal_idx}(:,:,k)=Coh_hat_sep_alt_rw(:,:,k)*Soh_hat_sep_alt_rw(:,:,k)';
            end

            fprintf(['        P group lasso\n'])
            So_hat_sep_pgl_colsp = zeros(O,O,K);
            P_hat_sep_pgl_colsp = zeros(O,O,K);
            for k=1:K
                [So_hat_sep_pgl_colsp(:,:,k),P_hat_sep_pgl_colsp(:,:,k)] = ...
                    estA_pgl_colsp(Co_est(:,:,k),N-O,regs);
            end
            So_sep_pgl_colsp{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_sep_pgl_colsp;
            P_sep_pgl_colsp{graph_trial_idx,signal_trial_idx,num_signal_idx} = P_hat_sep_pgl_colsp;

            fprintf(['        P group lasso with RW\n'])
            So_hat_sep_pgl_colsp_rw = zeros(O,O,K);
            P_hat_sep_pgl_colsp_rw = zeros(O,O,K);
            for k=1:K
                [So_hat_sep_pgl_colsp_rw(:,:,k),P_hat_sep_pgl_colsp_rw(:,:,k)] = ...
                    estA_pgl_colsp_rw(Co_est(:,:,k),N-O,regs);
            end
            So_sep_pgl_colsp_rw{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_sep_pgl_colsp_rw;
            P_sep_pgl_colsp_rw{graph_trial_idx,signal_trial_idx,num_signal_idx} = P_hat_sep_pgl_colsp_rw;

            fprintf(['        P nuc. norm\n'])
            So_hat_sep_pnn_colsp = zeros(O,O,K);
            P_hat_sep_pnn_colsp = zeros(O,O,K);
            for k=1:K
                [So_hat_sep_pnn_colsp(:,:,k),P_hat_sep_pnn_colsp(:,:,k)] = ...
                    estA_pnn_colsp(Co_est(:,:,k),N-O,regs);
            end
            So_sep_pnn_colsp{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_sep_pnn_colsp;
            P_sep_pnn_colsp{graph_trial_idx,signal_trial_idx,num_signal_idx} = P_hat_sep_pnn_colsp;

            fprintf(['        P nuc. norm with RW\n'])
            So_hat_sep_pnn_colsp_rw = zeros(O,O,K);
            P_hat_sep_pnn_colsp_rw = zeros(O,O,K);
            for k=1:K
                [So_hat_sep_pnn_colsp_rw(:,:,k),P_hat_sep_pnn_colsp_rw(:,:,k)] = ...
                    estA_pnn_colsp_rw(Co_est(:,:,k),N-O,regs);
            end
            So_sep_pnn_colsp_rw{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_sep_pnn_colsp_rw;
            P_sep_pnn_colsp_rw{graph_trial_idx,signal_trial_idx,num_signal_idx} = P_hat_sep_pnn_colsp_rw;


%             for k=1:K
%                 So_sep{graph_trial_idx,signal_trial_idx,num_signal_idx}=So;
%                 P_sep{graph_trial_idx,signal_trial_idx,num_signal_idx}=P;
%             end

            %------------------------

            %------------------------
            % Joint inference:
            fprintf(['        Joint inference\n'])

            fprintf(['        Alter. Min.\n'])
            [So_hat_alt,Soh_hat_alt,Coh_hat_alt] = ...
                estA_alt(Co_est,N-O,regs);
            So_joi_alt{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_alt;
            P_joi_alt{graph_trial_idx,signal_trial_idx,num_signal_idx} = zeros(O,O,K);
            for k=1:K
                P_joi_alt{graph_trial_idx,signal_trial_idx,num_signal_idx}(:,:,k)=Coh_hat_alt(:,:,k)*Soh_hat_alt(:,:,k)';
            end

            fprintf(['        Alter. Min. with RW\n'])
            [So_hat_alt_rw,Soh_hat_alt_rw,Coh_hat_alt_rw] = ...
                estA_alt_rw(Co_est,N-O,regs);
            So_joi_alt_rw{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_alt_rw;
            P_joi_alt_rw{graph_trial_idx,signal_trial_idx,num_signal_idx} = zeros(O,O,K);
            for k=1:K
                P_joi_alt_rw{graph_trial_idx,signal_trial_idx,num_signal_idx}(:,:,k)=Coh_hat_alt_rw(:,:,k)*Soh_hat_alt_rw(:,:,k)';
            end

            fprintf(['        P group lasso\n'])
            [So_hat_pgl_colsp,P_hat_pgl_colsp] = ...
                estA_pgl_colsp(Co_est,N-O,regs);
            So_joi_pgl_colsp{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_pgl_colsp;
            P_joi_pgl_colsp{graph_trial_idx,signal_trial_idx,num_signal_idx} = P_hat_pgl_colsp;

            fprintf(['        P group lasso with RW\n'])
            [So_hat_pgl_colsp_rw,P_hat_pgl_colsp_rw] = ...
                estA_pgl_colsp_rw(Co_est,N-O,regs);
            So_joi_pgl_colsp_rw{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_pgl_colsp_rw;
            P_joi_pgl_colsp_rw{graph_trial_idx,signal_trial_idx,num_signal_idx} = P_hat_pgl_colsp_rw;

            fprintf(['        P nuc. norm\n'])
            [So_hat_pnn_colsp,P_hat_pnn_colsp] = ...
                estA_pnn_colsp(Co_est,N-O,regs);
            So_joi_pnn_colsp{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_pnn_colsp;
            P_joi_pnn_colsp{graph_trial_idx,signal_trial_idx,num_signal_idx} = P_hat_pnn_colsp;

            fprintf(['        P nuc. norm with RW\n'])
            [So_hat_pnn_colsp_rw,P_hat_pnn_colsp_rw] = ...
                estA_pnn_colsp_rw(Co_est,N-O,regs);
            So_joi_pnn_colsp_rw{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_pnn_colsp_rw;
            P_joi_pnn_colsp_rw{graph_trial_idx,signal_trial_idx,num_signal_idx} = P_hat_pnn_colsp_rw;

%             for k=1:K
%                 So_joi{graph_trial_idx,signal_trial_idx,num_signal_idx}=So;
%                 P_joi{graph_trial_idx,signal_trial_idx,num_signal_idx}=P;
%             end
            %------------------------

            fprintf('\n')

        end
    end
end

figure();
g=1;s=1;r=1;
for k=1:K
    subplot(K,3,3*(k-1)+1);
    imagesc(So_true{g}(:,:,k));
    colorbar();
    title(['True graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+2);
    imagesc(So_sep_alt{g,s,r}(:,:,k));
    colorbar();
    title(['Sep. graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+3);
    imagesc(So_joi_alt{g,s,r}(:,:,k));
    colorbar();
    title(['Joi. graph ',num2str(k)]);
end

figure();
g=1;s=1;r=1;
for k=1:K
    subplot(K,3,3*(k-1)+1);
    imagesc(So_true{g}(:,:,k));
    colorbar();
    title(['True graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+2);
    imagesc(So_sep_alt_rw{g,s,r}(:,:,k));
    colorbar();
    title(['Sep. graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+3);
    imagesc(So_joi_alt_rw{g,s,r}(:,:,k));
    colorbar();
    title(['Joi. graph ',num2str(k)]);
end

figure();
g=1;s=1;r=1;
for k=1:K
    subplot(K,3,3*(k-1)+1);
    imagesc(So_true{g}(:,:,k));
    colorbar();
    title(['True graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+2);
    imagesc(So_sep_pgl_colsp{g,s,r}(:,:,k));
    colorbar();
    title(['Sep. graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+3);
    imagesc(So_joi_pgl_colsp{g,s,r}(:,:,k));
    colorbar();
    title(['Joi. graph ',num2str(k)]);
end

figure();
g=1;s=1;r=1;
for k=1:K
    subplot(K,3,3*(k-1)+1);
    imagesc(So_true{g}(:,:,k));
    colorbar();
    title(['True graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+2);
    imagesc(So_sep_pgl_colsp_rw{g,s,r}(:,:,k));
    colorbar();
    title(['Sep. graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+3);
    imagesc(So_joi_pgl_colsp_rw{g,s,r}(:,:,k));
    colorbar();
    title(['Joi. graph ',num2str(k)]);
end

figure();
g=1;s=1;r=1;
for k=1:K
    subplot(K,3,3*(k-1)+1);
    imagesc(So_true{g}(:,:,k));
    colorbar();
    title(['True graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+2);
    imagesc(So_sep_pnn_colsp{g,s,r}(:,:,k));
    colorbar();
    title(['Sep. graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+3);
    imagesc(So_joi_pnn_colsp{g,s,r}(:,:,k));
    colorbar();
    title(['Joi. graph ',num2str(k)]);
end

figure();
g=1;s=1;r=1;
for k=1:K
    subplot(K,3,3*(k-1)+1);
    imagesc(So_true{g}(:,:,k));
    colorbar();
    title(['True graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+2);
    imagesc(So_sep_pnn_colsp_rw{g,s,r}(:,:,k));
    colorbar();
    title(['Sep. graph ',num2str(k)]);
    subplot(K,3,3*(k-1)+3);
    imagesc(So_joi_pnn_colsp_rw{g,s,r}(:,:,k));
    colorbar();
    title(['Joi. graph ',num2str(k)]);
end


error_sep_trials_alt = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_sep_trials_alt_rw = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_sep_trials_pgl_colsp = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_sep_trials_pgl_colsp_rw = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_sep_trials_pnn_colsp = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_sep_trials_pnn_colsp_rw = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);

error_joi_trials_alt = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_joi_trials_alt_rw = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_joi_trials_pgl_colsp = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_joi_trials_pgl_colsp_rw = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_joi_trials_pnn_colsp = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_joi_trials_pnn_colsp_rw = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);

for g=1:num_graph_trials
    for s=1:num_signal_trials
        for r=1:num_signal_range
            for k=1:K
                So_sep_norm_alt           = So_sep_alt{g,s,r}(:,:,k)/sum(So_sep_alt{g,s,r}(:,1,k));
                So_sep_norm_alt_rw        = So_sep_alt_rw{g,s,r}(:,:,k)/sum(So_sep_alt_rw{g,s,r}(:,1,k));
                So_sep_norm_pgl_colsp     = So_sep_pgl_colsp{g,s,r}(:,:,k)/sum(So_sep_pgl_colsp{g,s,r}(:,1,k));
                So_sep_norm_pgl_colsp_rw  = So_sep_pgl_colsp_rw{g,s,r}(:,:,k)/sum(So_sep_pgl_colsp_rw{g,s,r}(:,1,k));
                So_sep_norm_pnn_colsp     = So_sep_pnn_colsp{g,s,r}(:,:,k)/sum(So_sep_pnn_colsp{g,s,r}(:,1,k));
                So_sep_norm_pnn_colsp_rw  = So_sep_pnn_colsp_rw{g,s,r}(:,:,k)/sum(So_sep_pnn_colsp_rw{g,s,r}(:,1,k));

                So_joi_norm_alt           = So_joi_alt{g,s,r}(:,:,k)/sum(So_joi_alt{g,s,r}(:,1,k));
                So_joi_norm_alt_rw        = So_joi_alt_rw{g,s,r}(:,:,k)/sum(So_joi_alt_rw{g,s,r}(:,1,k));
                So_joi_norm_pgl_colsp     = So_joi_pgl_colsp{g,s,r}(:,:,k)/sum(So_joi_pgl_colsp{g,s,r}(:,1,k));
                So_joi_norm_pgl_colsp_rw  = So_joi_pgl_colsp_rw{g,s,r}(:,:,k)/sum(So_joi_pgl_colsp_rw{g,s,r}(:,1,k));
                So_joi_norm_pnn_colsp     = So_joi_pnn_colsp{g,s,r}(:,:,k)/sum(So_joi_pnn_colsp{g,s,r}(:,1,k));
                So_joi_norm_pnn_colsp_rw  = So_joi_pnn_colsp_rw{g,s,r}(:,:,k)/sum(So_joi_pnn_colsp_rw{g,s,r}(:,1,k));

                So_true_norm = So_true{g}(:,:,k)/sum(So_true{g}(:,1,k));

                error_sep_trials_alt(g,s,r,k) = ...
                    norm(So_sep_norm_alt(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_sep_trials_alt_rw(g,s,r,k) = ...
                    norm(So_sep_norm_alt_rw(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_sep_trials_pgl_colsp(g,s,r,k) = ...
                    norm(So_sep_norm_pgl_colsp(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_sep_trials_pgl_colsp_rw(g,s,r,k) = ...
                    norm(So_sep_norm_pgl_colsp_rw(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_sep_trials_pnn_colsp(g,s,r,k) = ...
                    norm(So_sep_norm_pnn_colsp(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_sep_trials_pnn_colsp_rw(g,s,r,k) = ...
                    norm(So_sep_norm_pnn_colsp_rw(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;

                error_joi_trials_alt(g,s,r,k) = ...
                    norm(So_joi_norm_alt(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_joi_trials_alt_rw(g,s,r,k) = ...
                    norm(So_joi_norm_alt_rw(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_joi_trials_pgl_colsp(g,s,r,k) = ...
                    norm(So_joi_norm_pgl_colsp(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_joi_trials_pgl_colsp_rw(g,s,r,k) = ...
                    norm(So_joi_norm_pgl_colsp_rw(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_joi_trials_pnn_colsp(g,s,r,k) = ...
                    norm(So_joi_norm_pnn_colsp(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_joi_trials_pnn_colsp_rw(g,s,r,k) = ...
                    norm(So_joi_norm_pnn_colsp_rw(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
            end
        end
    end
end

error_joi_alt = squeeze(mean(mean(mean(error_joi_trials_alt,1),2),3));
error_joi_alt_rw = squeeze(mean(mean(mean(error_joi_trials_alt_rw,1),2),3));
error_joi_pgl_colsp = squeeze(mean(mean(mean(error_joi_trials_pgl_colsp,1),2),3));
error_joi_pgl_colsp_rw = squeeze(mean(mean(mean(error_joi_trials_pgl_colsp_rw,1),2),3));
error_joi_pnn_colsp = squeeze(mean(mean(mean(error_joi_trials_pnn_colsp,1),2),3));
error_joi_pnn_colsp_rw = squeeze(mean(mean(mean(error_joi_trials_pnn_colsp_rw,1),2),3));

error_sep_alt = squeeze(mean(mean(mean(error_sep_trials_alt,1),2),3));
error_sep_alt_rw = squeeze(mean(mean(mean(error_sep_trials_alt_rw,1),2),3));
error_sep_pgl_colsp = squeeze(mean(mean(mean(error_sep_trials_pgl_colsp,1),2),3));
error_sep_pgl_colsp_rw = squeeze(mean(mean(mean(error_sep_trials_pgl_colsp_rw,1),2),3));
error_sep_pnn_colsp = squeeze(mean(mean(mean(error_sep_trials_pnn_colsp,1),2),3));
error_sep_pnn_colsp_rw = squeeze(mean(mean(mean(error_sep_trials_pnn_colsp_rw,1),2),3));

disp('Alt. Min.')
disp(['Joint err: ' num2str(error_joi_alt')])
disp(['Separ err: ' num2str(error_sep_alt')])
disp(['Mean joint err: ' num2str(mean(error_joi_alt))])
disp(['Mean separ err: ' num2str(mean(error_sep_alt))])

disp('Alt. Min. with RW')
disp(['Joint err: ' num2str(error_joi_alt_rw')])
disp(['Separ err: ' num2str(error_sep_alt_rw')])
disp(['Mean joint err: ' num2str(mean(error_joi_alt_rw))])
disp(['Mean separ err: ' num2str(mean(error_sep_alt_rw))])

disp('P group lasso')
disp(['Joint err: ' num2str(error_joi_pgl_colsp')])
disp(['Separ err: ' num2str(error_sep_pgl_colsp')])
disp(['Mean joint err: ' num2str(mean(error_joi_pgl_colsp))])
disp(['Mean separ err: ' num2str(mean(error_sep_pgl_colsp))])

disp('P group lasso with RW')
disp(['Joint err: ' num2str(error_joi_pgl_colsp_rw')])
disp(['Separ err: ' num2str(error_sep_pgl_colsp_rw')])
disp(['Mean joint err: ' num2str(mean(error_joi_pgl_colsp_rw))])
disp(['Mean separ err: ' num2str(mean(error_sep_pgl_colsp_rw))])

disp('P nuc. norm')
disp(['Joint err: ' num2str(error_joi_pnn_colsp')])
disp(['Separ err: ' num2str(error_sep_pnn_colsp')])
disp(['Mean joint err: ' num2str(mean(error_joi_pnn_colsp))])
disp(['Mean separ err: ' num2str(mean(error_sep_pnn_colsp))])

disp('P nuc. norm with RW')
disp(['Joint err: ' num2str(error_joi_pnn_colsp_rw')])
disp(['Separ err: ' num2str(error_sep_pnn_colsp_rw')])
disp(['Mean joint err: ' num2str(mean(error_joi_pnn_colsp_rw))])
disp(['Mean separ err: ' num2str(mean(error_sep_pnn_colsp_rw))])

error_joi_alt_plot = reshape(mean(mean(error_joi_trials_alt,1),2),num_signal_range,K);
error_joi_alt_rw_plot = reshape(mean(mean(error_joi_trials_alt_rw,1),2),num_signal_range,K);
error_joi_pgl_colsp_plot = reshape(mean(mean(error_joi_trials_pgl_colsp,1),2),num_signal_range,K);
error_joi_pgl_colsp_rw_plot = reshape(mean(mean(error_joi_trials_pgl_colsp_rw,1),2),num_signal_range,K);
error_joi_pnn_colsp_plot = reshape(mean(mean(error_joi_trials_pnn_colsp,1),2),num_signal_range,K);
error_joi_pnn_colsp_rw_plot = reshape(mean(mean(error_joi_trials_pnn_colsp_rw,1),2),num_signal_range,K);

error_sep_alt_plot = reshape(mean(mean(error_sep_trials_alt,1),2),num_signal_range,K);
error_sep_alt_rw_plot = reshape(mean(mean(error_sep_trials_alt_rw,1),2),num_signal_range,K);
error_sep_pgl_colsp_plot = reshape(mean(mean(error_sep_trials_pgl_colsp,1),2),num_signal_range,K);
error_sep_pgl_colsp_rw_plot = reshape(mean(mean(error_sep_trials_pgl_colsp_rw,1),2),num_signal_range,K);
error_sep_pnn_colsp_plot = reshape(mean(mean(error_sep_trials_pnn_colsp,1),2),num_signal_range,K);
error_sep_pnn_colsp_rw_plot = reshape(mean(mean(error_sep_trials_pnn_colsp_rw,1),2),num_signal_range,K);

colors = {'#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'};

figure();
semilogx(signal_range,mean(error_joi_alt_plot,2),         'o-','Color',colors{1});
hold on;
semilogx(signal_range,mean(error_joi_alt_rw_plot,2),      'o-','Color',colors{2});
semilogx(signal_range,mean(error_joi_pgl_colsp_plot,2),   'o-','Color',colors{3});
semilogx(signal_range,mean(error_joi_pgl_colsp_rw_plot,2),'o-','Color',colors{4});
semilogx(signal_range,mean(error_joi_pnn_colsp_plot,2),   'o-','Color',colors{5});
semilogx(signal_range,mean(error_joi_pnn_colsp_rw_plot,2),'o-','Color',colors{6});

semilogx(signal_range,mean(error_sep_alt_plot,2),         's:','Color',colors{1});
semilogx(signal_range,mean(error_sep_alt_rw_plot,2),      's:','Color',colors{2});
semilogx(signal_range,mean(error_sep_pgl_colsp_plot,2),   's:','Color',colors{3});
semilogx(signal_range,mean(error_sep_pgl_colsp_rw_plot,2),'s:','Color',colors{4});
semilogx(signal_range,mean(error_sep_pnn_colsp_plot,2),   's:','Color',colors{5});
semilogx(signal_range,mean(error_sep_pnn_colsp_rw_plot,2),'s:','Color',colors{6});

hold off;
xlabel('Number of signals','Interpreter','Latex')
ylabel('Recovery error','Interpreter','Latex')
grid on
legend('J AM','J AM with RW','J PGL','J PGL with RW','J PNN','J PNN with RW',...
    'S AM','S AM with RW','S PGL','S PGL with RW','S PNN','S PNN with RW')
% legend('Joint 1','Separate 1',...
%        'Joint 2','Separate 2',...
%        'Joint 3','Separate 3')
% axis([signal_range(1) signal_range(end) ...
%     min(min(error_joi_trials(:)),min(error_sep_trials(:)))*.9 ...
%     max(max(error_joi_trials(:)),max(error_sep_trials(:)))*1.1])
% saveas(gcf,'???.jpg')
