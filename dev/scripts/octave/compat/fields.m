function f = fields(s)
%FIELDS The field names of a structure, as MATLAB's FIELDS returns them.
%   ibslike.m calls FIELDS (line 124 at 2229c00), which MATLAB keeps as an
%   alias of FIELDNAMES and Octave lacks.
f = fieldnames(s);
end
