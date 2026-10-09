function r = scripted_fun(x, dmat)
%SCRIPTED_FUN Simulator that returns the next entries of a fixed stream.
%   R = SCRIPTED_FUN(X, DMAT) ignores X and returns, as a column, the next
%   SIZE(DMAT, 1) entries of the global STREAM, from the global position
%   POS, which it advances.
global STREAM POS
n = size(dmat, 1);
r = STREAM(POS+1:POS+n);
r = r(:);
POS = POS + n;
end
