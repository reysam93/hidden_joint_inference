load('results/data_exp1_v1.mat');
nG = size(A_T,1);
nK = size(A_T,2);
MM = size(A_T,3);
HH = size(A_T,4);
kk=2;
aux1 = A_T{1,1,1,1};
aux2 = A_T{1,kk,1,1};
K1 = size(aux1,4);
K2 = size(aux2,4);


res_joint1 = zeros(nG,MM,HH,K1);
res_sep1 = zeros(nG,MM,HH,K1);
res_joint2 = zeros(nG,MM,HH,K2);
res_sep2 = zeros(nG,MM,HH,K2);
mdl = 'fronorm';%fscore %Qlinks
mtrc = 'Mean';
fmts = {'s:','x:','o:','*:','s-','x-','o-','*-'};
%models = {'No hidden','PNN no-sim-P','PGL','PNN'};

for m = 1:MM
    for hd = 1:HH
        for g = 1:nG
            A_aux1 = A_T{g,1,m,hd};%A_T{g,m,1,hd};
            [res_joint1(g,m,hd,:),res_sep1(g,m,hd,:)] = get_results(A_aux1,mdl); 
            A_aux2 = A_T{g,kk,m,hd};%A_T{g,m,2,hd};
            [res_joint2(g,m,hd,:),res_sep2(g,m,hd,:)] = get_results(A_aux2,mdl); 
        end
    end
end

if strcmp(mtrc,'Mean')
    %res_err_sep1 = squeeze(mean(mean(res_sep1,4)));
    res_err_joint1 = squeeze(mean(mean(res_joint1,4)));
    %res_err_sep2 = squeeze(mean(mean(res_sep2,4)));
    res_err_joint2 = squeeze(mean(mean(res_joint2,4)));
elseif strcmp(mtrc,'Median')
    %res_err_sep1 = squeeze(median(mean(res_sep1,4)));
    res_err_joint1 = squeeze(median(mean(res_joint1,4)));
    %res_err_sep2 = squeeze(median(mean(res_sep2,4)));
    res_err_joint2 = squeeze(median(mean(res_joint2,4)));
elseif strcmp(mtrc,'Recovery median')
    res_err_joint1 = squeeze(sum(median(res_joint1,4)==1)/nG);
    res_err_joint2 = squeeze(sum(median(res_joint2,4)==1)/nG);   
elseif strcmp(mtrc,'Recovery mean')
    res_err_joint1 = squeeze(sum(mean(res_joint1,4)==1)/nG);
    res_err_joint2 = squeeze(sum(mean(res_joint2,4)==1)/nG); 
end

% Plot properties
mark_s = 8;
line_w = 2;

% Mean error
set(0,'defaultTextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

figure()
MM = [1,3,4];
for t = MM
    plot(1:HH,res_err_joint1(t,:),fmts{t},'MarkerSize',mark_s,'LineWidth',line_w)
    hold on
    %lgnd{t} = ['K=3 ' models{t}];
end

for t = MM
    plot(1:HH,res_err_joint2(t,:),fmts{t+4},'MarkerSize',mark_s,'LineWidth',line_w)
    hold on
    %lgnd{t+MM} = ['K=6 ' models{t}];
end
lgnd = {'No hidden, K=3','PGL, K=3','PNN, K=3',...
    'No hidden, K=6','PGL, K=6','PNN, K=6',};
% for t = 1:MM
%     plot(1:HH,res_err_sep(t,:),fmts{t+4},'MarkerSize',12,'LineWidth',2)
%     hold on
%     lgnd{t+MM} = ['Sep ' models{t}];
% end
legend(lgnd,'Location','southeast')
if strcmp(mdl,'fronorm')
    ylabel('Mean error')
    yticks(0:.1:.7)
    ylim([0 .7])
%     title([mtrc ' of the Frobenius norm'])
else 
%     title([mtrc ' of the Fscore'])
    if strcmp(mtrc,'Recovery')
        ylabel('Fraction of recovered graphs')
        title(mtrc)
    end
end
xlabel('(a) Number of hidden variables')
grid on;
set(gca,'FontSize',16);
set(gcf, 'PaperPositionMode', 'auto')