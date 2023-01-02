function [As] =  gen_similar_graphs(A,K,pert_links)
    N = size(A,1);
    As = zeros(N,N,K);
    As(:,:,1) = A;
    for k=2:K
        As(:,:,k) = A;
        for i=1:pert_links
            node_id = randi(N);

            % delete link
            [link_nodes,~] = find(As(:,node_id,1)~=0);
            del_node = link_nodes(randperm(length(link_nodes),1));
            As(node_id,del_node,k) = 0;
            As(del_node,node_id,k) = 0;

            % create link
            [nonlink_nodes,~] = find(As(:,node_id,1)==0);
            nonlink_nodes(nonlink_nodes==node_id) = [];  % avoid self loops
            add_node = nonlink_nodes(randperm(length(nonlink_nodes),1));
            As(node_id,add_node,k) = 1;
            As(add_node,node_id,k) = 1;
        end
        assert(sum(sum(As(:,:,k)~=As(:,:,k)'))==0,'Non-symmetric matrix')
        assert(sum(diag(As(:,:,k)))==0,'Non-zero diagonal')
    end
end
