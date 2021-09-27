function [As] = get_student_networks_graphs(graphs_idx,N,path)
% graphs_idx needs to be a vector with values between 1 and 12 

if nargin < 4
    path = 'data/student_networks/';
end

K = length(graphs_idx);

% Load social networks
G = cell(K,1);
for k=1:K
    file = [path 'as' num2str(graphs_idx(k)) '.net.txt'];
    G{k} = load(file);
end

% Read graph
As = zeros(N,N,K);
for k=1:K
    for idx=1:size(G{k},1)
        i = G{k}(idx,1);
        j = G{k}(idx,2); 
        l = G{k}(idx,3);
        As(j,i,k) = l;
        As(i,j,k) = l;
    end
end