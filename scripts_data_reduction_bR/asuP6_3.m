function hkl_asu=asuP6_3(hkl)
  % Apply symmetry transformations and Friedel transformation so that 
  % all reflections are in the asymmetric unit. - Cecilia C, - 18 March 2019
  h = hkl(1);
  k = hkl(2);
  l = hkl(3);
  switch ((h<0)*8+(k<0)*4+(l<0)*2+(h+k<0))
    case {0,1}
    % (h>=0),(k>=0),(l>=0),(h+k>=0)
    % (h>=0),(k>=0),(l>=0),(h+k<0)
      hkl_asu = [h,k,l];                                                       % Real space (x, y, z) - Cecilia C. 18 March 2019
    case {2,3}
    % (h>=0),(k>=0),(l<0),(h+k>=0)
    % (h>=0),(k>=0),(l<0),(h+k<0)
      hkl_asu = [h,k,-l];                                                      % Real space (-x, -y, 1/2+z) - Cecilia C. 18 March 2019
    case 4
    % (h>=0),(k<0),(l>=0),(h+k>=0)
      hkl_asu = [-k,h+k,l];                                                    % Real space (x-y, x, 1/2+z) - Cecilia C. 18 March 2019
    case 5
    % (h>=0),(k<0),(l>=0),(h+k<0)
      hkl_asu = [-h-k,h,l];                                                    % Real space (-y, x-y, z) - Cecilia C. 18 March 2019
    case 6
    % (h>=0),(k<0),(l<0),(h+k>=0)
      hkl_asu = [-k,h+k,-l];                                                   % Real space (-x+y, -x, z) - Cecilia C. 18 March 2019
    case 7
    % (h>=0),(k<0),(l<0),(h+k<0)
      hkl_asu = [-h-k,h,-l];                                                   % Real space (y, -x+y, 1/2+z) - Cecilia C. 18 March 2019
    case 8
    % (h<0),(k>=0),(l>=0),(h+k>=0)
      hkl_asu = [h+k,-h,l];                                                    % Real space (y, -x+y, 1/2+z) - Cecilia C. 18 March 2019
    case 9
    % (h<0),(k>=0),(l>=0),(h+k<0)
      hkl_asu = [k,-h-k,l];                                                    % Real space (-x+y, -x, z) - Cecilia C. 18 March 2019
    case 10
    % (h<0),(k>=0),(l<0),(h+k>=0)
      hkl_asu = [h+k,-h,-l];                                                   % Real space (-y, x-y, z) - Cecilia C. 18 March 2019
    case 11
    % (h<0),(k>=0),(l<0),(h+k<0)
      hkl_asu = [k,-h-k,-l];                                                   % Real space (x-y, x, 1/2+z) - Cecilia C. 18 March 2019
    case {12,13}
    % (h<0),(k<0),(l>=0),(h+k>=0)
    % (h<0),(k<0),(l>=0),(h+k<0)
      hkl_asu = [-h,-k,l];                                                     % Real space (-x, -y, 1/2+z) - Cecilia C. 18 March 2019
    case {14,15}
    % (h<0),(k<0),(l<0),(h+k>=0)
    % (h<0),(k<0),(l<0),(h+k<0)
      hkl_asu = [-h,-k,-l];                                                    % Real space (x, y, z) - Cecilia C. 18 March 2019
  end
%end function hkl_asu