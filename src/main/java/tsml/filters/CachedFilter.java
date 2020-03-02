package tsml.filters;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

/**
 * Cache the filtering operation using a map. Note, the instances must be hashed first to use the cache reliably
 * otherwise issues occur with instance copying changing the hashcode due to memory locations.
 * <p>
 * Contributors: goastler
 */
public class CachedFilter extends HashFilter {

    private Filter filter;
    private Map<Instance, Instance> cache;
    private Queue<Instance> transformQueue = new LinkedList<>();

    public CachedFilter(Filter filter) {
        this(filter, new HashMap<>());
    }

    public CachedFilter(Filter filter, Map<Instance, Instance> cache) {
        setFilter(filter);
        setCache(cache);
    }

    public CachedFilter(CachedFilter other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    public Capabilities getCapabilities() {
        return filter.getCapabilities();
    }

    public boolean setInputFormat(Instances instanceInfo) throws Exception {
        super.setInputFormat(instanceInfo);
        boolean outputFormatHasBeenSetup = filter.setInputFormat(instanceInfo);
        return outputFormatHasBeenSetup;
    }

    public boolean batchFinished() throws Exception {
        if(getInputFormat() == null) {
            throw new NullPointerException("No input instance format defined");
        }
        filter.batchFinished();
        Instances outputFormat = filter.getOutputFormat();
        if(outputFormat != null) {
            setOutputFormat(outputFormat);
        }
        flushInput();
        m_NewBatch = true;
        m_FirstBatchDone = true;
        return (numPendingOutput() != 0);
    }

    @Override
    public Instance output() {
        Instance output = super.output();
        if(output == null) {
            if(transformQueue.isEmpty()) {
                // then nothing left to output
                return null;
            }
            // then the filter has the output
            Instance transformed = filter.output();
            Instance instance = transformQueue.remove();
            cache.put(instance, transformed);
            output = transformed;
        }
        return output;
    }

    @Override
    public boolean input(Instance instance) throws Exception {
        if(!(instance instanceof HashFilter.HashedDenseInstance)) {
            throw new IllegalArgumentException("can only handle hashed instances for reliable hashing");
        }
        Instance transformed = cache.get(instance);
        if(transformed == null) {
            // no cached copy so input to the filter
            boolean processed = filter.input(instance);
            // has the instance been instantly processed
            if(processed) {
                // yes, then grab the transformed instance
                transformed = filter.output();
                // add it back to the cache
                cache.put(instance, transformed);
                // add it to the output queue
                push(transformed);
                return true;
            } else {
                // no, instance is waiting for processing
                // null signifies that the instance has been passed to the filter for transformation
                // so when encountering nulls on the output func we know to get the instance from the filter instead
                push(null);
                // add the raw instance to the transform queue
                transformQueue.add(instance);
            }
            return processed;
        } else {
            // cached copy so add to processing queue as nothing needs to be done
            push(transformed);
            return true;
        }
    }

    public Filter getFilter() {
        return filter;
    }

    public void setFilter(Filter filter) {
        if(filter == null) {
            throw new NullPointerException();
        }
        this.filter = filter;
    }

    public Map<Instance, Instance> getCache() {
        return cache;
    }

    public void setCache(Map<Instance, Instance> cache) {
        if(cache == null) {
            throw new NullPointerException();
        }
        this.cache = cache;
    }
}
