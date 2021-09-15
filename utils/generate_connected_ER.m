function G = generate_connected_ER(N, p)
% This function generates a connected ER graph from parameters N and p
% N: number of nodes
% p: connection probability
flag_connected = 0;
while flag_connected ==0
    G = rand(N,N) < p;
    G = triu(G,1);
    G = G + G'; % Adjacency matrix
    % Check that graph is connected
    L = diag(sum(G))-G;
    [~, Lambda] = eig(L);
    if sum(diag(abs(Lambda))<=10^-6)==1
        flag_connected = 1;
    end
end
end