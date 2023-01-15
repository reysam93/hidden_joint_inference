function [Cs] = create_cov(As,L,M,sampled,type,h)
    if nargin < 5
        type = 'poly';
    end
    if nargin < 6
        h = rand(L,1)*2;
        % h = randn(L,1);
        h = h/norm(h,1);
    end

    % Create covariances
    N = size(As,1);
    K = size(As,3);
    Cs = zeros(size(As));
    for k=1:K
        if strcmp(type,'poly')
            H = zeros(N);
            for l=1:L
                H = H + h(l)*As(:,:,k)^(l-1);
            end
            C_true = H^2;

        elseif strcmp(type,'mrf')
            eigvals = eig(As(:,:,k));
            C_inv = (0.01-min(eigvals))*eye(N,N) + (0.9+0.1*rand(1,1))*As(:,:,k);
            C_true = inv(C_inv);
        else
            error('ERR: Unknown covariance type')
        end

        if sampled
            X = sqrtm(C_true)*randn(N,M);
            Cs(:,:,k) = X*X'/(M-1);
        else
            Cs(:,:,k) = C_true;
        end
    end
end