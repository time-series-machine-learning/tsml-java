package tsml.transformers;

import java.util.HashMap;
import java.util.Map;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: cache the filtering operation using a map. Note, the instances must be hashed first to use the cache
 * reliably otherwise issues occur with instance copying changing the hashcode due to memory locations.
 * <p>
 * Contributors: goastler, abostrom
 */
public class CachedTransformer extends BaseTrainableTransformer {

    // the filter to cache the output of
    private Transformer transformer;

    // the cache to store instances against their corresponding transform output
    private Map<Instance, Instance> cache;

    public CachedTransformer(Transformer transformer) {
        setTransformer(transformer);
        reset();
    }

    public void reset() {
        super.reset();
        cache = null;
    }

    @Override
    public void fit(final Instances data) {
        super.fit(data);
        // make the cache match the size of the data (as that is the max expected cache entries at any point in time)
        // . Load factor of 1 should mean if no more than data size instances are added, the hashmap will not expand
        // and waste cpu time
        cache = new HashMap<>(data.size(), 1);
    }

    @Override
    public String toString() {
        return transformer.getClass().getSimpleName();
    }

    public Capabilities getCapabilities() {
        return transformer.getCapabilities();
    }

    public void setTransformer(final Transformer transformer) {
        Assert.assertNotNull(transformer);
        this.transformer = transformer;
    }

    public void setCache(final Map<Instance, Instance> cache) {
        Assert.assertNotNull(cache);
        this.cache = cache;
    }

    @Override
    public Instance transform(Instance instance) {
        if(!isFit()) {
            throw new IllegalStateException("must be fitted first");
        }
        Instance transformedInstance = cache.get(instance);
        // if not instance is present in cache then transform and add
        if(transformedInstance == null) {
            transformedInstance = transformer.transform(instance);
            cache.put(instance, transformedInstance);
        }
        return transformedInstance;
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

    @Override
    public void setParams(final ParamSet paramSet) throws Exception {
        super.setParams(paramSet);
        ParamHandlerUtils.setParam(paramSet, TRANSFORMER_FLAG, this::setTransformer, Transformer.class);
    }

    @Override
    public ParamSet getParams() {
        return new ParamSet().add(TRANSFORMER_FLAG, transformer);
    }

    public static final String TRANSFORMER_FLAG = "f";
}
