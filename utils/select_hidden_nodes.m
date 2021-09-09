function [n_o, n_h] = select_hidden_nodes(sel_type, O, A, X)
    if nargin < 4
        X = [];
    end
    
    N = size(A,1);
    n_o = [];
    n_h = [];

    switch sel_type
        % select nodes with less links
        case 'min' 
            connections = sum(A);
            [~,node_pos] = sort(connections,'descend');
        % select nodes with more links
        case 'max'
            connections = sum(A);
            [~,node_pos] = sort(connections,'ascend');
        % select nodes which when removed the smoothness is smaller
        case 'min_smooth'
            L = diag(sum(A))-A;
            smooth = node_smoothness(N, L, X);
            [~,node_pos] = sort(smooth,'descend');
        % select nodes which when removed the smoothness is bigger
        case 'max_smooth'
            L = diag(sum(A))-A;
            smooth = node_smoothness(N, L, X);
            [~,node_pos] = sort(smooth,'ascend');
        % select nodes which when removed the variation of the smoothness
        % is bigger
        case 'max_diff_sm'
            L = diag(sum(A))-A;
            smooth = node_smoothness(N, L, X);
            diff_sm = abs(smooth-trace(X'*L*X));
            [~,node_pos] = sort(diff_sm,'ascend');
        case 'rand'
            [~,node_pos] = sort(rand(N,1));
        % select nodes at random
        otherwise
            disp('ERR: unknown method for selecting the hidden node')
            return
    end
    n_o = sort(node_pos(1:O));
    n_h = sort(node_pos(O+1:N));
end


function smoothness = node_smoothness(N, L, X)
    smoothness = zeros(N,1);
    for i=1:N
        Lo = L([1:i-1 i+1:N], [1:i-1 i+1:N]);
        Xo = X([1:i-1 i+1:N], :);
        smoothness(i) = trace(Xo'*Lo*Xo);
    end
end