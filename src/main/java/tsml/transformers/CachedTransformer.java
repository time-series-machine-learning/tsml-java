package tsml.transformers;

import java.util.HashMap;
import java.util.Map;

import tsml.data_containers.TimeSeriesInstance;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: cache the filtering operation using a map. Note, the instances must
 * be hashed first to use the cache reliably otherwise issues occur with
 * instance copying changing the hashcode due to memory locations.
 * <p>
 * Contributors: goastler, abostrom
 */
public class CachedTransformer implements Transformer {

    // the filter to cache the output of
    private Transformer transformer;

    private HashTransformer hasher = new HashTransformer();

    // the cache to store instances against their corresponding output
    private Map<Instance, Instance> cache;
    private Map<TimeSeriesInstance, TimeSeriesInstance> ts_cache;

    public CachedTransformer(Transformer transformer) {
        this(transformer, new HashMap<>(), new HashMap<>());
    }

    public CachedTransformer(Transformer transformer, Map<Instance, Instance> cache, Map<TimeSeriesInstance, TimeSeriesInstance> ts_cache) {
        this.transformer = transformer;
        this.cache = cache;
        this.ts_cache = ts_cache;
    }

    @Override
    public String toString() {
        return transformer.getClass().getSimpleName();
    }

    public Capabilities getCapabilities() {
        return transformer.getCapabilities();
    }

    @Override
    public Instance transform(Instance inst) {

        // hash the instnace before we check whether it is in the map or not.
        inst = hasher.transform(inst);

        // if the key is not in the map, transform and store it.
        if (!cache.containsKey(inst)) {
            cache.put(inst, transformer.transform(inst));
        }

        return cache.get(inst);
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        // hash the instnace before we check whether it is in the map or not.
        inst = hasher.transform(inst);

        // if the key is not in the map, transform and store it.
        if (!ts_cache.containsKey(inst)) {
            ts_cache.put(inst, transformer.transform(inst));
        }

        return ts_cache.get(inst);
    }



    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        return transformer.determineOutputFormat(data);
    }

    public Transformer getTransformer() {
        return transformer;
    }

    public Map<Instance, Instance> getCache() {
        return cache;
    }



}
