function f = fields(s)
%FIELDS The field names of a structure, as MATLAB's FIELDS returns them.
%   MATLAB keeps FIELDS as an alias of FIELDNAMES, which ibslike.m calls
%   (line 124 at 2229c00). Octave has FIELDNAMES only.
f = fieldnames(s);
end
