function [precision,recall,f,NMI,num_of_edges] = graph_learning_perf_eval(A_0,A)
% evaluate the performance of graph learning algorithms

if sum(isnan(A))>0
    precision = 0;
    recall = 0;
    f = 0;
    NMI = 0;
    num_of_edges = 0;
    return
end

A_0tmp = A_0-diag(diag(A_0));
edges_groundtruth = squareform(A_0tmp)~=0;

Atmp = A-diag(diag(A));
edges_learned = squareform(Atmp)~=0;


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