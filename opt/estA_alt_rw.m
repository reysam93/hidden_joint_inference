function [A] = estA_alt_rw(Co,regs)
A = [];

O = size(Co,1);
K = size(Co,3);


cvx_begin quiet
    variable Ao(O,O,K) symmetric
    
cvx_end