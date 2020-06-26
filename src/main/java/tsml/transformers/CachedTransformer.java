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

    public boolean isCacheFittedDataOnly() {
        return cacheFittedDataOnly;
    }

    public void setCacheFittedDataOnly(final boolean cacheFittedDataOnly) {
        this.cacheFittedDataOnly = cacheFittedDataOnly;
    }

    public static class TransformedInstance {
        private TransformedInstance(final Instance instance) {
            this.instance = instance;
        }

        private Instance instance;

        public Instance getInstance() {
            return instance;
        }

        public void setInstance(final Instance instance) {
            this.instance = instance;
        }
    }

    // the filter to cache the output of
    private Transformer transformer;
    // whether to only cache instances from the fit() call OR all instances handed to the transform method
    private boolean cacheFittedDataOnly;

    // the cache to store instances against their corresponding transform output
    private Map<Instance, TransformedInstance> cache;
//    private final Indexer indexer = new Indexer();

    public CachedTransformer(Transformer transformer) {
        setTransformer(transformer);
        setCacheFittedDataOnly(true);
        reset();
    }

    public void reset() {
        super.reset();
        cache = null;
//        indexer.reset();
    }

    @Override
    public void fit(final Instances data) {
        super.fit(data);
//        indexer.fit(data);
        // make the cache match the size of the data (as that is the max expected cache entries at any point in time)
        // . Load factor of 1 should mean if no more than data size instances are added, the hashmap will not expand
        // and waste cpu time
        cache = new HashMap<>(data.size(), 1);
        for(Instance instance : data) {
            cache.put(instance, new TransformedInstance(null));
        }
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

    public void setCache(final Map<Instance, TransformedInstance> cache) {
        Assert.assertNotNull(cache);
        this.cache = cache;
    }

    @Override
    public Instance transform(Instance instance) {
        if(!isFit()) {
            throw new IllegalStateException("must be fitted first");
        }
//        if(!(instance instanceof Indexer.IndexedInstance)) {
//            return transformer.transform(instance);
//        }
        TransformedInstance transformedInstance = cache.get(instance);
        Instance transform;
        if(transformedInstance == null) {
            transform = transformer.transform(instance);
            if(!cacheFittedDataOnly) {
                cache.put(instance, new TransformedInstance(transform));
            }
        } else {
            transform = transformedInstance.getInstance();
            if(transform == null) {
                transform = transformer.transform(instance);
                transformedInstance.setInstance(transform);
            }
        }
        return transform;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        return transformer.determineOutputFormat(data);
    }

    public Transformer getTransformer() {
        return transformer;
    }

    public Map<Instance, TransformedInstance> getCache() {
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
