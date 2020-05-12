package tsml.filters;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

import java.util.Arrays;

import static utilities.Utilities.toInstances;

public class Utilities {
    public static Instances filter(Instances data, Filter filter) throws
                                                                  Exception {
        filter.setInputFormat(new Instances(data, 0));
        return Filter.useFilter(data, filter);
    }

    public static Instance filter(Instance instance, Filter filter) throws
                                                                    Exception {
        Instances instances = toInstances(instance);
        Instances filtered = filter(instances, filter);
        return filtered.get(0);
    }

}
