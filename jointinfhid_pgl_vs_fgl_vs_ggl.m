clear; clc;
addpath(['../../../Code/CVX/cvx'])
addpath(genpath('utils'));
% addpath(genpath('opt'));
rng(1000);
%--------------------------------------------------------------------------

%------------------------
num_graph_trials = 1;
num_signal_trials = 1;
%------------------------

%------------------------
signal_range = round(logspace(2,5,3));
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
PMRF_true = cell(num_graph_trials,1);
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
regs.lambda1  = 1e-2;
regs.lambda2  = 1e-2;
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
So_pgl = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_pgl = cell(num_graph_trials,num_signal_trials,num_signal_range);
So_fgl = cell(num_graph_trials,num_signal_trials,num_signal_range);
So_ggl = cell(num_graph_trials,num_signal_trials,num_signal_range);

So_pgl_MRF = cell(num_graph_trials,num_signal_trials,num_signal_range);
P_pgl_MRF = cell(num_graph_trials,num_signal_trials,num_signal_range);
So_fgl_MRF = cell(num_graph_trials,num_signal_trials,num_signal_range);
So_ggl_MRF = cell(num_graph_trials,num_signal_trials,num_signal_range);
%------------------------

%------------------------
error_trials_pgl = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_trials_fgl = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_trials_ggl = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);

error_trials_pgl_MRF = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_trials_fgl_MRF = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_trials_ggl_MRF = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
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
        end
        S_true{graph_trial_idx}(:,:,k) = S(:,:,k);
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
    
	CMRF = zeros(N,N,K);
    alpha = 5;
    for k=1:K
        CMRF(:,:,k) = inv(alpha*eye(N) + S(:,:,k));
    end
    %------------------------

    %------------------------
    % Select hidden nodes
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, S(:,:,1));
    So  = S(n_o,n_o,:);
    Soh = S(n_o,n_h,:);
    
    Co  = C(n_o,n_o,:);
    Coh = C(n_o,n_h,:);
    CMRFo  = CMRF(n_o,n_o,:);
    CMRFoh = CMRF(n_o,n_h,:);
    
    So_true{graph_trial_idx}  = S_true{graph_trial_idx}(n_o,n_o,:);
    Soh_true{graph_trial_idx} = S_true{graph_trial_idx}(n_o,n_h,:);
    %------------------------

    %------------------------
    % P matrices
    P_true{graph_trial_idx} = zeros(O,O,K);
    for k=1:K
        P_true{graph_trial_idx}(:,:,k) = Coh(:,:,k)*(Soh(:,:,k)');
    end
    
    PMRF_true{graph_trial_idx} = zeros(O,O,K);
    for k=1:K
        PMRF_true{graph_trial_idx}(:,:,k) = CMRFoh(:,:,k)*(Soh(:,:,k)');
    end
    %------------------------

    %------------------------
    % Commutativity check
    comm = norm(vec(pagemtimes(C,S)-pagemtimes(S,C)),'fro')^2;
    comm_obs = norm(vec(pagemtimes(Co,So)-pagemtimes(So,Co)),'fro')^2;
    disp(['norm(CA-AC) = ' num2str(comm) '  -  norm(CoAo-AoCo) = '...
        num2str(comm_obs)])

    comm = norm(vec(pagemtimes(CMRF,S)-pagemtimes(S,CMRF)),'fro')^2;
    comm_obs = norm(vec(pagemtimes(CMRFo,So)-pagemtimes(So,CMRFo)),'fro')^2;
    disp(['norm(CA-AC) = ' num2str(comm) '  -  norm(CoAo-AoCo) = '...
        num2str(comm_obs),'    (MRF)'])
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
        
        XMRF_MAX = cell(K,1);
        for k=1:K
            XMRF_MAX{k} = mvnrnd(zeros(N,1),CMRF(:,:,k),signal_range(end))';
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

            XMRF = cell(K,1);
            for k=1:K
                XMRF{k} = XMRF_MAX{k}(:,1:num_signals);
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
            % Sample covariance
            CMRF_est = zeros(N,N,K);
            for k=1:K
                CMRF_est(:,:,k)=XMRF{k}*XMRF{k}'/num_signals;
            end
            CMRFo_est = CMRF_est(n_o,n_o,:);
            %------------------------

            %------------------------
            % PGL (Poly):
            fprintf(['        Joint inference PGL (Poly) \n'])
            [So_hat_pgl,P_hat_pgl] = ...
                estA_pgl_colsp_rw(Co_est,N-O,regs);
            So_pgl{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_pgl;
            P_pgl{graph_trial_idx,signal_trial_idx,num_signal_idx} =P_hat_pgl;

            %------------------------

            %------------------------
            % PGL (MRF):
            fprintf(['        Joint inference PGL (MRF)\n'])
            [So_hat_pgl,P_hat_pgl] = ...
                estA_pgl_colsp_rw(CMRFo_est,N-O,regs);
            So_pgl_MRF{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_pgl;
            P_pgl_MRF{graph_trial_idx,signal_trial_idx,num_signal_idx} =P_hat_pgl;
            %------------------------
            
            %------------------------
            % FGL (Poly):
            fprintf(['        Joint inference FGL (Poly)\n'])
            So_hat_fgl = ...
                fglasso(Co_est,regs);
            So_fgl{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_fgl;
            %------------------------

            %------------------------
            % FGL (MRF):
            fprintf(['        Joint inference FGL (MRF)\n'])
            So_hat_fgl_MRF = ...
                fglasso(CMRFo_est,regs);
            So_fgl_MRF{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_fgl_MRF;
            %------------------------
            
            %------------------------
            % GGL (Poly):
            fprintf(['        Joint inference GGL (Poly)\n'])
            So_hat_ggl = ...
                gglasso(Co_est,regs);
            So_ggl{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_ggl;
            %------------------------

            %------------------------
            % GGL (MRF):
            fprintf(['        Joint inference GGL (MRF)\n'])
            So_hat_ggl_MRF = ...
                gglasso(CMRFo_est,regs);
            So_ggl_MRF{graph_trial_idx,signal_trial_idx,num_signal_idx}=So_hat_ggl_MRF;
            %------------------------
            
            fprintf('\n')

        end
    end
end

figure();
g=1;s=1;r=1;
for k=1:K
    subplot(K,3,3*(k-1)+1);
    imagesc(So_true{g}(:,:,k)/sum(So_true{g}(:,1,k)));
    colorbar();
    subplot(K,3,3*(k-1)+2);
    imagesc(So_pgl{g,s,r}(:,:,k));
    colorbar();
    title(['PGL graph ',num2str(k),' (poly)']);
    subplot(K,3,3*(k-1)+3);
    imagesc(So_pgl_MRF{g,s,r}(:,:,k));
    colorbar();
    title(['PGL graph ',num2str(k),' (MRF)']);
end

figure();
g=1;s=1;r=1;
for k=1:K
    subplot(K,3,3*(k-1)+1);
    imagesc(So_true{g}(:,:,k)/sum(So_true{g}(:,1,k)));
    colorbar();
    subplot(K,3,3*(k-1)+2);
    imagesc(So_fgl{g,s,r}(:,:,k));
    colorbar();
    title(['FGL graph ',num2str(k),' (poly)']);
    subplot(K,3,3*(k-1)+3);
    imagesc(So_fgl_MRF{g,s,r}(:,:,k));
    colorbar();
    title(['FGL graph ',num2str(k),' (MRF)']);
end

figure();
g=1;s=1;r=1;
for k=1:K
    subplot(K,3,3*(k-1)+1);
    imagesc(So_true{g}(:,:,k)/sum(So_true{g}(:,1,k)));
    colorbar();
    subplot(K,3,3*(k-1)+2);
    imagesc(So_ggl{g,s,r}(:,:,k));
    colorbar();
    title(['GGL graph ',num2str(k),' (poly)']);
    subplot(K,3,3*(k-1)+3);
    imagesc(So_ggl_MRF{g,s,r}(:,:,k));
    colorbar();
    title(['GGL graph ',num2str(k),' (MRF)']);
end

for g=1:num_graph_trials
    for s=1:num_signal_trials
        for r=1:num_signal_range
            for k=1:K
                So_norm_pgl = So_pgl{g,s,r}(:,:,k)/sum(So_pgl{g,s,r}(:,1,k));
                So_norm_fgl = So_fgl{g,s,r}(:,:,k);
                So_norm_ggl = So_ggl{g,s,r}(:,:,k);
%                 So_norm_fgl = So_fgl{g,s,r}(:,:,k)/sum(So_fgl{g,s,r}(:,1,k));
%                 So_norm_ggl = So_ggl{g,s,r}(:,:,k)/sum(So_ggl{g,s,r}(:,1,k));

                So_norm_pgl_MRF = So_pgl_MRF{g,s,r}(:,:,k)/sum(So_pgl_MRF{g,s,r}(:,1,k));
                So_norm_fgl_MRF = So_fgl_MRF{g,s,r}(:,:,k);
                So_norm_ggl_MRF = So_ggl_MRF{g,s,r}(:,:,k);
%                 So_norm_fgl_MRF = So_fgl_MRF{g,s,r}(:,:,k)/sum(So_fgl_MRF{g,s,r}(:,1,k));
%                 So_norm_ggl_MRF = So_ggl_MRF{g,s,r}(:,:,k)/sum(So_ggl_MRF{g,s,r}(:,1,k));

                So_true_norm = So_true{g}(:,:,k)/sum(So_true{g}(:,1,k));

                error_trials_pgl(g,s,r,k) = ...
                    norm(So_norm_pgl(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_trials_pgl_MRF(g,s,r,k) = ...
                    norm(So_norm_pgl_MRF(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;

                error_trials_fgl(g,s,r,k) = ...
                    norm(So_norm_fgl(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_trials_fgl_MRF(g,s,r,k) = ...
                    norm(So_norm_fgl_MRF(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;

                error_trials_ggl(g,s,r,k) = ...
                    norm(So_norm_ggl(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                error_trials_ggl_MRF(g,s,r,k) = ...
                    norm(So_norm_ggl_MRF(:) - So_true_norm(:),'fro')^2/norm(So_true_norm(:),'fro')^2;
                
            end
        end
    end
end

error_pgl = squeeze(mean(mean(mean(error_trials_pgl,1),2),3));
error_fgl = squeeze(mean(mean(mean(error_trials_fgl,1),2),3));
error_ggl = squeeze(mean(mean(mean(error_trials_ggl,1),2),3));

error_pgl_MRF = squeeze(mean(mean(mean(error_trials_pgl_MRF,1),2),3));
error_fgl_MRF = squeeze(mean(mean(mean(error_trials_fgl_MRF,1),2),3));
error_ggl_MRF = squeeze(mean(mean(mean(error_trials_ggl_MRF,1),2),3));

disp(['PGL err (Poly): ' num2str(error_pgl')])
disp(['FGL err (Poly): ' num2str(error_fgl')])
disp(['GGL err (Poly): ' num2str(error_ggl')])

disp(['Mean PGL err (Poly): ' num2str(mean(error_pgl))])
disp(['Mean FGL err (Poly): ' num2str(mean(error_fgl))])
disp(['Mean GGL err (Poly): ' num2str(mean(error_ggl))])

disp(['PGL err (MRF): ' num2str(error_pgl')])
disp(['FGL err (MRF): ' num2str(error_fgl')])
disp(['GGL err (MRF): ' num2str(error_ggl')])

disp(['Mean PGL err (MRF): ' num2str(mean(error_pgl))])
disp(['Mean FGL err (MRF): ' num2str(mean(error_fgl))])
disp(['Mean GGL err (MRF): ' num2str(mean(error_ggl))])

error_pgl_plot = reshape(mean(mean(error_trials_pgl,1),2),num_signal_range,K);
error_fgl_plot = reshape(mean(mean(error_trials_fgl,1),2),num_signal_range,K);
error_ggl_plot = reshape(mean(mean(error_trials_ggl,1),2),num_signal_range,K);
error_pgl_MRF_plot = reshape(mean(mean(error_trials_pgl_MRF,1),2),num_signal_range,K);
error_fgl_MRF_plot = reshape(mean(mean(error_trials_fgl_MRF,1),2),num_signal_range,K);
error_ggl_MRF_plot = reshape(mean(mean(error_trials_ggl_MRF,1),2),num_signal_range,K);

colors = {'#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'};

figure();
semilogx(signal_range,mean(error_pgl_plot,2),'o-','Color','red');
hold on;
semilogx(signal_range,mean(error_fgl_plot,2),'o-','Color','blue');
semilogx(signal_range,mean(error_ggl_plot,2),'o-','Color','green');

semilogx(signal_range,mean(error_pgl_MRF_plot,2),'s:','Color','red');
semilogx(signal_range,mean(error_fgl_MRF_plot,2),'s:','Color','blue');
semilogx(signal_range,mean(error_ggl_MRF_plot,2),'s:','Color','green');

hold off;
xlabel('Number of signals','Interpreter','Latex')
ylabel('Recovery error','Interpreter','Latex')
grid on
legend('PGL (Poly)','FGL (Poly)','GGL (Poly)',...
       'PGL (MRF)','FGL (MRF)','GGL (MRF)')

