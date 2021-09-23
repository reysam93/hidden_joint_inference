nG = size(A_T,1);
MM = size(A_T,2);
HH = size(A_T,3);
aux = A_T{1,1,1};
K = size(aux,4);

res_joint = zeros(nG,MM,HH,K);
res_sep = zeros(nG,MM,HH,K);
mdl = 'fscore';%fscore %Qlinks
mtrc = 'median';
fmts = {'s-','x-','o-','*-','s:','x:','o:','*:'};

for m = 1: MM
    for hd = 1:HH
        for g = 1:nG
            A_aux = A_T{g,m,hd};
            [res_joint(g,m,hd,:),res_sep(g,m,hd,:)] = get_results(A_aux,mdl);  
        end
    end
end

if strcmp(mtrc,'mean')
    res_err_sep = squeeze(mean(mean(res_sep,4)));
    res_err_joint = squeeze(mean(mean(res_joint,4)));
elseif strcmp(mtrc,'median')
    res_err_sep = squeeze(median(median(res_sep,4)));
    res_err_joint = squeeze(median(median(res_joint,4)));
end
for t = 1:MM
    plot(1:HH,res_err_joint(t,:),fmts{t},'MarkerSize',12,'LineWidth',2)
    hold on
    lgnd{t} = ['Joint ' models{t}];
end
for t = 1:MM
    plot(1:HH,res_err_sep(t,:),fmts{t+4},'MarkerSize',12,'LineWidth',2)
    hold on
    lgnd{t+MM} = ['Sep ' models{t}];
end
legend(lgnd)
ylabel('Frobenius norm')
xlabel('Number of graphs')
title(['K=' num2str(K)])