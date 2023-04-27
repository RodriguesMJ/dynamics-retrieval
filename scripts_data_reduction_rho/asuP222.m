function hkl_asu=asuP222(hkl)
  % Apply symmetry transformations and Friedel transformation so that 
  % all reflections are in the asymmetric unit. 
  h = hkl(1);
  k = hkl(2);
  l = hkl(3);
  hkl_asu = [abs(h), abs(k), abs(l)]; % P222
end