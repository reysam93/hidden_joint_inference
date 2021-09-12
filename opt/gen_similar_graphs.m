function S = gen_similar_graphs(K,N,num_rewired,edge_prob)
    low_tri_ind = find(tril(ones(N,N))-eye(N));
%     edge_prob=.25;
%     frac_rewired = .3;
%     num_rewired = round(frac_rewired*nchoosek(N,2));
%     num_rewired = 3;
    
    s_orig = binornd(1,edge_prob,nchoosek(N,2),1);
    S = cell(K,1);
    
    for k=1:K
        S{k} = zeros(N,N);
        s = s_orig;
        is_edge=find(s);
        no_edge=find(~s);

        remove_edges=is_edge(randi(length(is_edge),num_rewired,1));
        add_edges=no_edge(randi(length(no_edge),num_rewired,1));
        s(add_edges)=s(remove_edges);
        
        S{k}(low_tri_ind)=s;
        S{k} = S{k}+S{k}';
        S{k} = S{k}/sum(S{k}(:,1));
    end
    
end