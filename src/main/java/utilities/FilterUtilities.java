package utilities;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

public class FilterUtilities {
    public static Instances filter(Instances data, Filter filter) throws
                                                                  Exception {
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }
}
