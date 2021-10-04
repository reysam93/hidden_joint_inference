%% Check influence of hidden nodes
rng(10)
addpath(genpath('utils'));
addpath(genpath('opt'));

n_graphs = 12;
Ks = [3 6];
N = 20;
Os = 19:-1:15;
p = 0.2;
pert_links = 3;
L = 3;
hid_nodes = 'rand';

max_iters = 20;
% Regs for AM
% H=1 K=3
regs1 = struct();
regs1.alpha = 1;
regs1.lambda = 1;
regs1.beta = 10;
regs1.eta = 10;
regs1.mu = 1e3;
regs1.gamma = 1;
regs1.delta1 = 1e-4;
regs1.delta2 = 1e-4;
regs1.epsilon = 1e-6;

regs2 = struct();
regs2.alpha = 1;
regs2.lambda = 1e6;
regs2.beta = 25;
regs2.eta = 25;
regs2.mu = 1e4;
regs2.delta1 = 1e-4;
regs2.delta2 = 1e-4;
regs2.gamma = 1;
regs2.epsilon = 1e-6;
tic
err = zeros(2*length(Ks),length(Os),n_graphs);
for g=1:n_graphs
    disp(['G: ' num2str(g)])
    % Create graphs and get hidden nodes
    A = generate_connected_ER(N,p);
    As = gen_similar_graphs(A,Ks(end),pert_links);
        
    err_g = zeros(2*length(Ks),length(Os));
    for i=1:length(Os)
        O = Os(i);
        [n_o, n_h] = select_hidden_nodes(hid_nodes, O, As(:,:,1));
        
        % Create covariances
        Cs = zeros(N,N,Ks(end));
        for k=1:Ks(end)
            h = rand(L,1)*2-1;
            H = zeros(N);
            for l=1:L
                H = H + h(l)*As(:,:,k)^(l-1);
            end
            Cs(:,:,k) = H^2;
            Cs(:,:,k) = Cs(:,:,k)/norm(Cs(:,:,k),'fro');
        end
        
        Ao = As(n_o,n_o,:);
        Aoh = As(n_o,n_h,:);
        Co = Cs(n_o,n_o,:);
        Coh = Cs(n_o,n_h,:);
        norm_Ao = norm(Ao(:,:,k),'fro')^2;
        
        for j=1:length(Ks)
            K = Ks(j);
%             [Ao_am_hid,~,~] = estA_alt_rw(Co(:,:,1:K),N-O,regs1,max_iters);
%             Ao_am_nohid = estA_alt_rw_nohid(Co(:,:,1:K),regs1,max_iters);
            Ao_am_hid = estA_pnn_colsp_rw(Co(:,:,1:K),N-O,regs1,max_iters);
%             Ao_am_nohid = estA_pnn_colsp_rw(Co(:,:,1:K),N-O,regs2,max_iters);
            Ao_am_hid = Ao_am_hid./max(max(Ao_am_hid));
            Ao_am_nohid = Ao_am_nohid./max(max(Ao_am_nohid));
            
            err_aux = zeros(2,K);
            for k=1:K
                err_aux(1,k) = norm(Ao(:,:,k)-Ao_am_hid(:,:,k),'fro')^2/norm_Ao;
                err_aux(2,k) = norm(Ao(:,:,k)-Ao_am_nohid(:,:,k),'fro')^2/norm_Ao;
            end
            err_g(2*j-1,i) = mean(err_aux(1,:));
            err_g(2*j,i) = mean(err_aux(2,:));
            disp(err_aux)
        end        
    end
    err(:,:,g) = err_g;
end
t = toc;
disp(['--- ' num2str(t/3600) ' hours ---'])
%% Plot results
mean_err = mean(err,3);
median_err = median(err,3);

figure()
plot(N-Os,mean_err(1,:),'-o');hold on
plot(N-Os,mean_err(2,:),'--o');hold on
plot(N-Os,mean_err(3,:),'-x');hold on
plot(N-Os,mean_err(4,:),'--x');hold on
legend({'AM rw,K=3', 'AM rw nohid,K=3','AM rw,K=6', 'AM rw nohid,K=6'})
grid on
axis tight
xlabel('H')
title('Mean err')

figure()
plot(N-Os,median_err(1,:),'-o');hold on
plot(N-Os,median_err(2,:),'--o');hold on
plot(N-Os,median_err(3,:),'-x');hold on
plot(N-Os,median_err(4,:),'--x');hold on
legend({'AM rw,K=3', 'AM rw nohid,K=3','AM rw,K=6', 'AM rw nohid,K=6'})
grid on
axis tight
xlabel('H')
title('Median err')