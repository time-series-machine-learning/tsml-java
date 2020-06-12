package tsml.transformers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
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
public class CachedTransformer implements TrainableTransformer {

    // the filter to cache the output of
    private Transformer transformer;
    private final Indexer indexer = new Indexer();
    private boolean isFit = false;

    // the cache to store instances against their corresponding transform output
    private List<Instance> cache;

    public CachedTransformer(Transformer transformer) {
        setTransformer(transformer);
    }

    @Override
    public boolean isFit() {
        return isFit;
    }

    @Override
    public void fit(final Instances data) {
        indexer.transform(data);
        isFit = true;
        cache = new ArrayList<>(data.size());
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
            return cache.get(((IndexedInstance) inst).getIndex());
        } else {
            // otherwise transform the unseen instance (this would be a test instance, i.e. not in cache / should not
            // be added to cache)
            return transformer.transform(inst);
        }
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

}
