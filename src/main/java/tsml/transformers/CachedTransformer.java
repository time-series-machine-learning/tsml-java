package tsml.transformers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
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
public class CachedTransformer implements ParamHandler, TrainableTransformer {

    // the filter to cache the output of
    private Transformer transformer;
    private final Indexer indexer = new Indexer();
    private boolean isFit;

    // the cache to store instances against their corresponding transform output
    private List<Instance> cache;

    public CachedTransformer(Transformer transformer) {
        setTransformer(transformer);
        reset();
    }



    public void reset() {
        isFit = false;
        cache = null;
        indexer.reset();
    }

    @Override
    public void fit(final Instances data) {
        indexer.fit(data);
        cache = new ArrayList<>(indexer.size());
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

    public void setCache(final List<Instance> cache) {
        Assert.assertNotNull(cache);
        this.cache = cache;
    }

    @Override
    public Instance transform(Instance inst) {
        if(inst instanceof IndexedInstance) {
            // the instance is from the fitted data so fetch from cache if possible
            final int index = ((IndexedInstance) inst).getIndex();
            if(index >= 0) {
                return cache.get(index);
            }
        }
        // otherwise transform the unseen instance (this would be a test instance, i.e. not in cache / should not
        // be added to cache)
        return transformer.transform(inst);
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        return transformer.determineOutputFormat(data);
    }

    public Transformer getTransformer() {
        return transformer;
    }

    public List<Instance> getCache() {
        return cache;
    }

    @Override
    public boolean isFit() {
        return isFit;
    }

    @Override
    public void setParams(final ParamSet paramSet) {
        ParamHandler.setParam(paramSet, TRANSFORMER_FLAG, this::setTransformer, Transformer.class);
    }

    @Override
    public ParamSet getParams() {
        return new ParamSet().add(TRANSFORMER_FLAG, transformer);
    }

    public static final String TRANSFORMER_FLAG = "f";
}
