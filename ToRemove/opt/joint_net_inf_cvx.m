clear; clc;
addpath(['../../../Code/CVX/cvx'])
addpath(['data_and_results/social/student_networks'])
%--------------------------------------------------------------------------

%------------------------
% Load social networks
G{1}  = load('as1.net.txt');
G{2}  = load('as2.net.txt');
G{3}  = load('as3.net.txt');
G{4}  = load('as4.net.txt');
G{5}  = load('as5.net.txt');
G{6}  = load('as6.net.txt');
G{7}  = load('as7.net.txt');
G{8}  = load('as8.net.txt');
G{9}  = load('as9.net.txt');
G{10} = load('as10.net.txt');
G{11} = load('as11.net.txt');
G{12} = load('as12.net.txt');
n1=7;n2=9;n3=12;
%------------------------

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
N = 50;
N = 32;
Nchoose2 = N*(N-1)/2;
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

%     S = gen_similar_graphs(K,N,3,.2);

    for idx=1:size(G{n1},1)
        i=G{n1}(idx,1); j=G{n1}(idx,2); l=G{n1}(idx,3);
        S{1}(j,i)=l; S{1}(i,j)=l;
    end
    for idx=1:size(G{n2},1)
        i=G{n2}(idx,1); j=G{n2}(idx,2); l=G{n2}(idx,3);
        S{2}(j,i)=l; S{2}(i,j)=l;
    end
    for idx=1:size(G{n3},1)
        i=G{n3}(idx,1); j=G{n3}(idx,2); l=G{n3}(idx,3);
        S{3}(j,i)=l; S{3}(i,j)=l;
    end
    S{1} = S{1}/sum(S{1}(:,1));
    S{2} = S{2}/sum(S{2}(:,1));
    S{3} = S{3}/sum(S{3}(:,1));
    
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
            alpha = ones(K,1);
            beta = 10*ones(nchoosek(K,2),1);

            %------------------------
            % Sample covariance
            C_est = cell(K,1);
            for k=1:K
                C_est{k}=cov(X{k}');
            end
            %------------------------

            %------------------------
            % Separate inference:
            fprintf(['        Separate inference\n'])

            % Find minumum feasible epsilon for all graphs
            e = cell(K,1);
            for k=1:K
                fprintf(['            Graph ',num2str(k),'\n'])
                e_min=0; e_max=10;
                e_MAX=10; last_e=nan;
                while isnan(last_e)
                    for t=1:20
                        e{k}=(e_min+e_max)/2;
                        cvx_begin quiet
                            variable S_separate(N,N) symmetric
                            minimize( norm(S_separate(:),1) )
                            subject to
                                norm(C_est{k}*S_separate-S_separate*C_est{k},'fro')<=e{k}
                                S_separate>=0
                                diag(S_separate)<=1e-6
                                abs(sum(S_separate(:,1))-1)<=1e-6
                        cvx_end
                        if (max(isnan(S_separate(:)))==0) && (max(isinf(S_separate(:)))==0)
                            last_e=e{k};
                            e_max=e{k};
                        else
                            e_min=e{k};
                        end
                    end
                    if isnan(last_e); e_min=0; e_MAX=e_MAX+5; e_max=e_MAX; end            
                end
                e{k}=last_e;

                S_separate=nan;
                while (max(isnan(S_separate(:))))
                    cvx_begin quiet
                        variable S_separate(N,N) symmetric
                        minimize( norm(S_separate(:),1) )
                        subject to
                            norm(S_separate*C_est{k}-C_est{k}*S_separate,'fro')<=e{k}
                            S_separate>=0
                            diag(S_separate)<=1e-6
                            abs(sum(S_separate(:,1))-1)<=1e-6
                    cvx_end
                    e{k}=e{k}+.02*e{k}*double(max(isnan(S_separate(:))));
                end

                % Save separately inferred graphs
                S_sep{graph_trial_idx,signal_trial_idx,num_signal_idx,k}=S_separate;
            end
            %------------------------

            %------------------------
            % Joint inference:
            fprintf(['        Joint inference\n'])
            S1=nan;
            while (max(isnan(S1(:))))
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Joint inference
                cvx_begin quiet
                cvx_precision best
                    variable S1(N,N) symmetric
                    variable S2(N,N) symmetric
                    variable S3(N,N) symmetric
                    minimize(    alpha(1)*norm(S1(:),1)+alpha(2)*norm(S2(:),1)+alpha(3)*norm(S3(:),1)+...
                        beta(1)*norm(S1(:)-S2(:),1)+beta(2)*norm(S1(:)-S3(:),1)+beta(3)*norm(S2(:)-S3(:),1) )
                    subject to
                        norm(S1*C_est{1}-C_est{1}*S1,'fro')<=e{1}
                        norm(S2*C_est{2}-C_est{2}*S2,'fro')<=e{2}
                        norm(S3*C_est{3}-C_est{3}*S3,'fro')<=e{3}
                        S1>=0
                        S2>=0
                        S3>=0
                        diag(S1)<=1e-3
                        diag(S2)<=1e-3
                        diag(S3)<=1e-3
                        abs(sum(S1(:,1))-1)<=1e-3
                        abs(sum(S2(:,1))-1)<=1e-3
                        abs(sum(S3(:,1))-1)<=1e-3
                cvx_end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Update epsilon if necessary
                e{1}=e{1}+.1*e{1}*double(max(isnan(S1(:))));
                e{2}=e{2}+.1*e{2}*double(max(isnan(S1(:))));
                e{3}=e{3}+.1*e{3}*double(max(isnan(S1(:))));
            end

            % Save jointly inferred graphs
            S_joi{graph_trial_idx,signal_trial_idx,num_signal_idx,1}=S1;
            S_joi{graph_trial_idx,signal_trial_idx,num_signal_idx,2}=S2;
            S_joi{graph_trial_idx,signal_trial_idx,num_signal_idx,3}=S3;
            fprintf('\n')
            %------------------------
        
        end
    end
end

error_joi_trials = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
error_sep_trials = zeros(num_graph_trials,num_signal_trials,num_signal_range,K);
for g=1:num_graph_trials
    for s=1:num_signal_trials
        for r=1:num_signal_range
            for k=1:K
                error_sep_trials(g,s,r,k) = ...
                    norm(S_sep{g,s,r,k}(:) - S_true{g,k}(:),'fro')/norm(S_true{g,k}(:),'fro');
                error_joi_trials(g,s,r,k) = ...
                    norm(S_joi{g,s,r,k}(:) - S_true{g,k}(:),'fro')/norm(S_true{g,k}(:),'fro');
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
