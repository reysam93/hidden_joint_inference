function [Cs] = create_cov(As,L,M,sampled)
    % Create covariances
    N = size(As,1);
    K = size(As,3);
    Cs = zeros(size(As));
    for k=1:K
        %h = rand(L,1)*2-1;
        h = randn(L,1);
        H = zeros(N);
        for l=1:L
            H = H + h(l)*As(:,:,k)^(l-1);
        end

        if sampled
            X = H*randn(N,M);
            Cs(:,:,k) = X*X'/M;
        else
            Cs(:,:,k) = H^2;
        end
    end
end