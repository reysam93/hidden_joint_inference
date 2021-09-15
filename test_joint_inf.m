clear; clc;
addpath(['../../../Code/CVX/cvx'])
% addpath(genpath('utils'));
% addpath(genpath('opt'));
%--------------------------------------------------------------------------

%------------------------
num_graph_trials = 1;
num_signal_trials = 3;
S_true = cell(num_graph_trials,1);
P_true = cell(num_graph_trials,1);
So_true = cell(num_graph_trials,1);
Soh_true = cell(num_graph_trials,1);
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

hid_nodes = 'min';
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
P_joi = cell(num_graph_trials,num_signal_trials,num_signal_range,K);
S_sep = cell(num_graph_trials,num_signal_trials,num_signal_range,K);
P_sep = cell(num_graph_trials,num_signal_trials,num_signal_range,K);
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

    for k=1:K
        So_true{graph_trial_idx,k}  = S_true{graph_trial_idx,k}(n_o,n_o);
        Soh_true{graph_trial_idx,k} = S_true{graph_trial_idx,k}(n_o,n_h);
    end
    %------------------------

    %------------------------
    % P matrices
    for k=1:K
        P_true{graph_trial_idx,k} = Coh{k}*(Soh{k}');
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
                    cost_sep = cost_sep + alpha*norm(vec(So(:,:,k)),1) + gamma*sum(norms(P(:,:,k)));
                end
                minimize(cost_sep)
                subject to
                    So >= 0;
%                     sum(So(:,1,1))== 1;
                    for k=1:K
                        sum(So(:,1,k))== 1;
                        diag(So(:,:,k)) == 0;
                        norm(Co_est{k}*So(:,:,k)+P(:,:,k)-So(:,:,k)*Co_est{k}-P(:,:,k)','fro') <= eps;
                    end
            cvx_end
            for k=1:K
                S_sep{graph_trial_idx,signal_trial_idx,num_signal_idx,k}=So(:,:,k);
                P_sep{graph_trial_idx,signal_trial_idx,num_signal_idx,k}=P(:,:,k);
            end

            %------------------------

            %------------------------
            % Joint inference:
            fprintf(['        Joint inference\n'])

            Q = cell(K,1);
            max_iters = 10;

            cvx_begin quiet
                variable So(O,O,K) symmetric
                variable P(O,O,K)
                variable Pp(O,O,K)
                variable Pm(O,O,K)

                cost_sep = 0;
                for k=1:K
                    cost_sep = cost_sep + alpha*norm(vec(So(:,:,k)),1) + gamma*sum(norms(P(:,:,k)));

                    for j=2:k
                       cost_sep = cost_sep + beta*norm(vec(So(:,:,k)-So(:,:,j)),1);

                       for i=1:O
                           cost_sep = cost_sep + eta*norm( Pp(:,i,k)+Pm(:,i,k) - Pp(:,i,j)-Pm(:,i,j) ,1);
                       end
                    end
                end
                minimize(cost_sep)
                subject to
                    So >= 0;
%                     sum(So(:,1,1))== 1;
                    P == Pp-Pm;
                    Pp>=0;
                    Pm>=0;
                    for k=1:K
                        sum(So(:,1,k))== 1;
                        diag(So(:,:,k)) == 0;
                        norm(Co_est{k}*So(:,:,k)+P(:,:,k)-So(:,:,k)*Co_est{k}-P(:,:,k)','fro') <= eps;
                    end
            cvx_end

            for k=1:K
                S_joi{graph_trial_idx,signal_trial_idx,num_signal_idx,k}=So(:,:,k);
                P_joi{graph_trial_idx,signal_trial_idx,num_signal_idx,k}=P(:,:,k);
            end
            %------------------------

        end
    end
end

fignum=1;

figure(fignum);fignum=fignum+1;
g=1;k=1;s=1;r=1;
subplot(1,3,1);
imagesc(S_true{g,k});
colorbar();
title(['True graph ',num2str(k)]);
subplot(1,3,2);
imagesc(S_sep{g,s,r,k});
colorbar();
title(['Sep. graph ',num2str(k)]);
subplot(1,3,3);
imagesc(S_joi{g,s,r,k});
colorbar();
title(['Joi. graph ',num2str(k)]);

figure(fignum);fignum=fignum+1;
g=1;k=1;s=1;r=1;
subplot(1,3,1);
imagesc(P_true{g,k});
colorbar();
title(['True P ',num2str(k)]);
subplot(1,3,2);
imagesc(P_sep{g,s,r,k});
colorbar();
title(['Sep. P ',num2str(k)]);
subplot(1,3,3);
imagesc(P_joi{g,s,r,k});
colorbar();
title(['Joi. P ',num2str(k)]);

% for k=1:K
%     subplot(1,K,k);
%     imagesc(S_true{k})
%     colorbar();
%     title(['Graph ',num2str(k)]);
% end

error_joi_trials = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_sep_trials = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
for g=1:num_graph_trials
    for s=1:num_signal_trials
        for r=1:num_signal_range
            for k=1:K
                error_sep_trials(g,s,r,k) = ...
                    norm(S_sep{g,s,r,k}(:) - So_true{g,k}(:),'fro')^2/norm(So_true{g,k}(:),'fro')^2;
                error_joi_trials(g,s,r,k) = ...
                    norm(S_joi{g,s,r,k}(:) - So_true{g,k}(:),'fro')^2/norm(So_true{g,k}(:),'fro')^2;
            end
        end
    end
end

error_joi = squeeze(mean(squeeze(mean(error_joi_trials,1)),1));
error_sep = squeeze(mean(squeeze(mean(error_sep_trials,1)),1));

colors = {'#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30'};
legend_names = [];

figure(fignum);fignum=fignum+1;
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
