package timeseriesweka.filters;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class CachedFilter extends SimpleBatchFilter {

    private final SimpleBatchFilter filter;
    private final HashMap<Integer, Instance> cache = new HashMap<>();

    public CachedFilter(final SimpleBatchFilter filter) {this.filter = filter;}


    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected Instances determineOutputFormat(final Instances inputFormat) throws
                                                                           Exception {
        throw new UnsupportedOperationException();
    }

    @Override
    protected Instances process(final Instances instances) throws
                                                           Exception {
        Instances results = new Instances(instances, 0);
        Instances toConvert = new Instances(instances, 0);
        for(Instance instance : instances) {
            double[] raw = instance.toDoubleArray();
            int hash = Arrays.hashCode(raw);
            Instance converted = cache.get(hash);
            if(converted == null) {
                toConvert.add(instance);
            } else {
                results.add(instance);
            }
        }
        Instances converted = Filter.useFilter(toConvert, filter);
        results.addAll(converted);
        for(Instance instance : converted) {
            double[] raw = instance.toDoubleArray();
            int hash = Arrays.hashCode(raw);
            cache.put(hash, instance);
        }
        return results;
    }
}
