package utilities;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

import java.util.Arrays;

public class FilterUtilities {
    public static Instances filter(Instances data, Filter filter) throws
                                                                  Exception {
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }

    public static Instance filter(Instance instance, Filter filter) throws
                                                                    Exception {
        Instances instances = toInstances(instance);
        Instances filtered = filter(instances, filter);
        return filtered.get(0);
    }

    public static Instances toInstances(Instance... instances) {
        Instances result = new Instances(instances[0].dataset(), 0);
        result.addAll(Arrays.asList(instances));
        return result;
    }
}
