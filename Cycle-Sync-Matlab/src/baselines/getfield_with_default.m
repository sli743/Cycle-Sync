function val = getfield_with_default(s,name,default)
if isfield(s,name)
    val = s.(name);
else
    val = default;
end
end
