function [Cs] = create_cov(As,prms)
    % Create covariances
    L = prms.L;
    M = prms.M;
    sampled = prms.sampled;
    Ctype = prms.Ctype;
    N = size(As,1);
    K = size(As,3);
    Cs = zeros(size(As));
    for k=1:K
        if strcmp(Ctype,'Cpoly')
            %h = rand(L,1)*2-1;
            h = randn(L,1);
            h = h/norm(h,1);
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
        elseif strcmp(Ctype,'Cmrf')
            eigvals = eig(As(:,:,k));
            C_inv = (0.01-min(eigvals))*eye(N,N) + (0.9+0.1*rand(1,1))*As(:,:,k);
            Cs(:,:,k) = inv(C_inv);
            if sampled
                W = randn(N,M);
                X(:,:,k) = sqrtm(Cs(:,:,k))*W;
                Cs(:,:,k) = X(:,1:M,k)*X(:,1:M,k)'/M;
            end
        else
            disp('Unknown covariance model')
        end
    end
end