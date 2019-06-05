package distances.derivative_time_domain;

import timeseriesweka.filters.CachedFilter;
import timeseriesweka.filters.DerivativeFilter;
import weka.filters.Filter;

public class Derivative {

    public static final Filter DERIVATIVE_FILTER = new CachedFilter(new DerivativeFilter());

}
