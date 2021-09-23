function [precision,recall,f,NMI,num_of_edges] = graph_learning_perf_eval(L_0,L)
% evaluate the performance of graph learning algorithms

if sum(isnan(L))>0
    precision = 0;
    recall = 0;
    f = 0;
    NMI = 0;
    num_of_edges = 0;
    return
end

L_0tmp = L_0-diag(diag(L_0));
edges_groundtruth = squareform(L_0tmp)~=0;

Ltmp = L-diag(diag(L));
edges_learned = squareform(Ltmp)~=0;


num_of_edges = sum(edges_learned);

if num_of_edges > 0
    [precision,recall] = perfcurve(double(edges_groundtruth),double(edges_learned),1,'Tvals',1,'xCrit','prec','yCrit','reca');
    if precision == 0 && recall == 0
        f = 0;
    else
        f = 2*precision*recall/(precision+recall);
    end
    NMI = perfeval_clus_nmi(double(edges_groundtruth),double(edges_learned));
else
    precision = 0;
    recall = 0;
    f = 0;
    NMI = 0;
end