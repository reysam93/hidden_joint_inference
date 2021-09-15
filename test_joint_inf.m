clear; clc;
addpath(['../../../Code/CVX/cvx'])
addpath(genpath('utils'));
addpath(genpath('opt'));
%--------------------------------------------------------------------------

%------------------------
num_graph_trials = 1;
num_signal_trials = 3;
S_true = cell(num_graph_trials,1);
%------------------------

%------------------------
signal_range = round(logspace(2,5,2));
num_signal_range = length(signal_range);
%------------------------

%--------------------------------------------------------------------------
% Set parameters

%------------------------
% Graph parameters
K = 3;
N = 20;
O = 19;
Nchoose2 = N*(N-1)/2;

hid_nodes = 'min'
pert_links = 3;
L = 3;
%------------------------

%------------------------
% Estimation parameters
alpha = 1;
beta = 1;
eta = 1;
lambd = 1;
mu = 1;
gamma = 1;
epsilon = 1e-6;
%------------------------

%------------------------
% Filter parameters
F = 3;
h = randn(F,1);
%------------------------

%------------------------
low_tri_ind = find(tril(ones(N,N))-eye(N));
%------------------------

%------------------------
S_joi = cell(num_graph_trials,num_signal_trials,num_signal_range,K);
S_sep = cell(num_graph_trials,num_signal_trials,num_signal_range,K);
%------------------------

%--------------------------------------------------------------------------
% GRAPH TRIALS:
for graph_trial_idx=1:num_graph_trials
    fprintf(['Graph trial ',num2str(graph_trial_idx),' of ',num2str(num_graph_trials),'\n'])

    %------------------------
    % Adjacency matrices
%     T = edge_prob*ones(N,N);
%     S = cell(K,1);
%     for k=1:K
%         S{k} = zeros(N,N);
%         S{k}(low_tri_ind)=binornd(1,edge_prob,Nchoose2,1);
%         S{k} = S{k} + S{k}';
%         S_true{graph_trial_idx,k} = S{k};
%     end

     S = gen_similar_graphs(K,N,3,.2);

    for k=1:K
        S_true{graph_trial_idx,k} = S{k};
    end

    %------------------------

    %------------------------
    % Graph filters
    H = cell(K,1);
    for k=1:K
        H{k} = zeros(N,N);
        for f=1:F
            H{k} = H{k} + h(f)*(S{k}^(f-1));
        end
    end
    %------------------------

    %------------------------
    % Covariance matrices
    C = cell(K,1);
    for k=1:K
        C{k} = H{k}^2;
    end
    %------------------------

    %------------------------
    % Select hidden nodes
    [n_o, n_h] = select_hidden_nodes(hid_nodes, O, S{1});
    So  = cell(K,1);
    Soh = cell(K,1);
    Co  = cell(K,1);
    Coh = cell(K,1);
    for k=1:K
        So{k}  = S{k}(n_o,n_o);
        Soh{k} = S{k}(n_o,n_h);
        Co{k}  = C{k}(n_o,n_o);
        Coh{k} = C{k}(n_o,n_h);
    end
    %------------------------

    %----------------------------------------------------------------------
    % SIGNAL TRIALS:
    for signal_trial_idx=1:num_signal_trials
        fprintf(['    Signal trial ',num2str(signal_trial_idx),' of ',num2str(num_signal_trials),'\n'])

        %------------------------
        X_MAX = cell(K,1);
        for k=1:K
            X_MAX{k} = H{k}*randn(N,signal_range(end));
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
            C_est = cell(K,1);
            Co_est = cell(K,1);
            for k=1:K
                C_est{k}=cov(X{k}');
                Co_est{k}  = C_est{k}(n_o,n_o);
            end
            %------------------------

            %------------------------
            % Separate inference:
            fprintf(['        Separate inference\n'])

            cvx_begin quiet
                variable So(O,O,K) symmetric
                variable P(O,O,K)

                cost_sep = 0;
                for k=1:K
                    cost_sep = cost_sep + alpha*norm(vec(So(:,:,k)),1) + gamma*sum(norms(P(:,:,k)))
                end
                minimize(cost_sep)
                subject to
                    norm(C_est{k}*S_o(:,:,k)-S_o(:,:,k)*C_est{k},'fro')<=epsilon
                    So >= 0;
                    sum(So(:,1,1))== 1;
                    for k=1:K
                        diag(So(:,:,k)) == 0;
                        norm(Co_est(:,:,k)*So(:,:,k)+P(:,:,k)-So(:,:,k)*Co_est(:,:,k)-P(:,:,k)','fro') <= eps;
                    end
            cvx_end
            %------------------------

            %------------------------
            % Joint inference:
            fprintf(['        Joint inference\n'])

            cvx_begin quiet
                variable So(O,O,K) symmetric
                variable P(O,O,K)

                cost_sep = 0;
                for k=1:K
                    cost_sep = cost_sep + alpha*norm(vec(So(:,:,k)),1) + gamma*sum(norms(P(:,:,k)))

                    norms_Pk = norms(P(:,:,k))
                    for j=2:k
                       cost_sep = cost_sep + beta*norm(vec(So(:,:,k)-So(:,:,j)),1);

                       norms_Pj = norms(P(:,:,j))
                       cost_sep = cost_sep + eta*norm(norms_Pk-norms_Pj,1)
                    end
                end
                minimize(cost_sep)
                subject to
                    norm(C_est{k}*S_o(:,:,k)-S_o(:,:,k)*C_est{k},'fro')<=epsilon
                    So >= 0;
                    sum(So(:,1,1))== 1;
                    for k=1:K
                        diag(So(:,:,k)) == 0;
                        norm(Co_est(:,:,k)*So(:,:,k)+P(:,:,k)-So(:,:,k)*Co_est(:,:,k)-P(:,:,k)','fro') <= eps;
                    end
            cvx_end
            So_est = cell(K,1)
            P_est = cell(K,1)
            for k=1:K
                So_est{k} = So(:,:,k)
                P_est{k} = P(:,:,k)
            end
            %------------------------

        end
    end
end

figure(1);
for k=1:K
    subplot(1,K,k);
    imagesc(S_true{k})
    colorbar();
    title(['Graph ',num2str(k)]);
end

error_joi_trials = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_sep_trials = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
for g=1:num_graph_trials
    for s=1:num_signal_trials
        for r=1:num_signal_range
            for k=1:K
                error_sep_trials(g,s,r,k) = ...
                    norm(S_sep{g,s,r,k}(:) - S_true{g,k}(:),'fro')^2/norm(S_true{g,k}(:),'fro')^2;
                error_joi_trials(g,s,r,k) = ...
                    norm(S_joi{g,s,r,k}(:) - S_true{g,k}(:),'fro')^2/norm(S_true{g,k}(:),'fro')^2;
            end
        end
    end
end

error_joi = squeeze(mean(squeeze(mean(error_joi_trials,1)),1));
error_sep = squeeze(mean(squeeze(mean(error_sep_trials,1)),1));

colors = {'#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30'};
legend_names = [];

figure(1)
for k=1:K
    semilogx(signal_range,error_joi(:,k),'o-','Color',colors{k})
    hold on
    semilogx(signal_range,error_sep(:,k),'s:','Color',colors{k})
end
hold off
xlabel('Number of signals','Interpreter','Latex')
ylabel('Recovery error','Interpreter','Latex')
grid on
legend('Joint 1','Separate 1',...
       'Joint 2','Separate 2',...
       'Joint 3','Separate 3')
axis([signal_range(1) signal_range(end) ...
    min(min(error_joi(:)),min(error_sep(:)))*.9 ...
    max(max(error_joi(:)),max(error_sep(:)))*1.1])
% saveas(gcf,'???.jpg')
