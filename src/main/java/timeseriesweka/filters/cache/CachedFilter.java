package timeseriesweka.filters.cache;

import weka.core.Instance;
import weka.filters.Filter;

import java.util.HashMap;

public class CachedFilter {
    private final Filter filter;
    private final HashMap<Instance, Instance> map = new HashMap<>();

    public CachedFilter(final Filter filter) {this.filter = filter;}

    public void get(Instance instance) {

    }
}
