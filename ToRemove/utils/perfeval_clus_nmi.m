function NMI = perfeval_clus_nmi(X,Y)
%   evaluate clustering X with respect to groundtruth Y
%   in terms of NORMALIZED MUTUAL INFORMATION
%   X is set of class index of length n
%   Y is set of cluster index of length n
%
%   inspired by Michael Chen's Information Theory Toolbox

n = numel(X);
[~,~,idxX] = unique(X); % idx for each entry
[~,~,idxY] = unique(Y); % idx for each entry

% joint distribution matrix of X and Y
% each column represents for each cluster its overlapping with classes
DistXY = accumarray([idxX(:) idxY(:)],1,[],[],[],true);

pXY = DistXY(:)/n;
pX = sum(DistXY,2)/n;
pY = sum(DistXY,1)/n;

% Mo Chen's version
% DistMatX = sparse(1:n,idxX,1,n,max(idxX),n);
% DistMatY = sparse(1:n,idxY,1,n,max(idxY),n);
% DistXY = DistMatX'*DistMatY;
% 
% pXY = DistXY(:)/n;
% pX = mean(DistMatX,1);
% pY = mean(DistMatY,1);

HXY = full( sum( -( pXY(pXY>0).*(log2(pXY(pXY>0))) ) ) ); % mutual information
HX = full( sum( -( pX(pX>0).*(log2(pX(pX>0))) ) ) ); % entropy of X
HY = full( sum( -( pY(pY>0).*(log2(pY(pY>0))) ) ) ); % entropy of Y

NMI = (HX+HY-HXY) / ((HX+HY)/2); % compute nmi