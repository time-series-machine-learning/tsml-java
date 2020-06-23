package tsml.transformers;

import java.util.HashMap;
import java.util.Map;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.transformers.Indexer.IndexedInstance;
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
    private final Indexer indexer = new Indexer();

    // the cache to store instances against their corresponding transform output
    private Map<IndexedInstance, IndexedInstance> cache;

    public CachedTransformer(Transformer transformer) {
        setTransformer(transformer);
        reset();
    }

    public void reset() {
        super.reset();
        cache = null;
        indexer.reset();
    }

    @Override
    public void fit(final Instances data) {
        super.fit(data);
        indexer.fit(data);
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

    public void setCache(final Map<IndexedInstance, IndexedInstance> cache) {
        Assert.assertNotNull(cache);
        this.cache = cache;
    }

    @Override
    public IndexedInstance transform(Instance instance) {
        if(!isFit()) {
            throw new IllegalStateException("must be fitted first");
        }
        IndexedInstance transformedInstance = null;
        int index = -1;
        if(instance instanceof IndexedInstance) {
            // the instance is from the fitted data so fetch from cache if possible
            index = ((IndexedInstance) instance).getIndex();
        }
        if(index >= 0) {
            // if index is non-negative then the instance is from the fitted data and is eligible for caching
            transformedInstance = cache.get(instance);
            // if not instance is present in cache then transform and add
            if(transformedInstance == null) {
                transformedInstance = new IndexedInstance(transformer.transform(instance), index);
                cache.put((IndexedInstance) instance, transformedInstance);
            }
        } else {
            // otherwise transform the unseen instance (this would be a test instance, i.e. not in cache / should not
            // be added to cache)
            transformedInstance = new IndexedInstance(transformer.transform(instance), -1);
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

    public Map<IndexedInstance, IndexedInstance> getCache() {
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
