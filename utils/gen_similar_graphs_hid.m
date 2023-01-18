function [As] =  gen_similar_graphs_hid(A,K,pert_links,n_o,n_h)
    N = size(A,1);
    As = zeros(N,N,K);

    Aos = gen_similar_graphs(A(n_o,n_o),K,pert_links);
    Aoh = A(n_o,n_h);
    A_h = A(n_h,n_h);

    As(:,:,1) = A;
    for k=2:K
        As(n_o,n_o,k) = Aos(:,:,k);
        As(n_o,n_h,k) = Aoh;
        As(n_h,n_o,k) = Aoh';
        As(n_h,n_h,k) = A_h;
    end
end
