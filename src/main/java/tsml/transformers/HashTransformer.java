package tsml.transformers;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.List;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

/**
 * Purpose: transforms a set of instances into a set of hashed instances. This
 * allows them to reliably be cached.
 * <p>
 * Contributors: goastler, abostrom
 */

public class HashTransformer implements Transformer {

    @Override
    public Instances determineOutputFormat(final Instances inputFormat) {
        return inputFormat;
    }

    @Override
    public Instances transform(Instances inst) {
        hashInstances(inst);
        return inst;
    }

    @Override
    public Instance transform(Instance inst) {
        return inst.dataset() != null ? hashInstanceAndDataset(inst) : hashInstance(inst);
    }

    public static class HashedDenseInstance extends DenseInstance {
        private int id;

        public HashedDenseInstance(Instance instance) {
            super(instance);
            setDataset(instance.dataset());
            if (instance instanceof HashedDenseInstance) {
                id = ((HashedDenseInstance) instance).id;
            } else {
                rebuildId();
            }
        }

        private void rebuildId() {
            id = Arrays.hashCode(m_AttValues);
        }

        @Override
        public int hashCode() {
            return id;
        }

        @Override
        public boolean equals(Object o) {
            boolean result = false;
            if (o instanceof HashedDenseInstance) {
                HashedDenseInstance other = (HashedDenseInstance) o;
                result = other.id == id;
            }
            return result;
        }

        @Override
        public Object copy() {
            return new HashedDenseInstance(this);
        }

        @Override
        protected void forceDeleteAttributeAt(int position) {
            rebuildId();
            super.forceDeleteAttributeAt(position);
        }

        @Override
        protected void forceInsertAttributeAt(int position) {
            rebuildId();
            super.forceInsertAttributeAt(position);
        }

        @Override
        public Instance mergeInstance(Instance inst) {
            rebuildId();
            return super.mergeInstance(inst);
        }
    }

    public static void hashInstances(List<Instance> instances) {
        for (int i = 0; i < instances.size(); i++) {
            Instance instance = instances.get(i);
            if (!(instance instanceof HashedDenseInstance)) {
                HashedDenseInstance indexedInstance = hashInstance(instance);
                instances.set(i, indexedInstance);
            }
        }
    }

    public static HashedDenseInstance hashInstance(Instance instance) {
        if (!(instance instanceof HashedDenseInstance)) {
            instance = new HashedDenseInstance(instance);
        }
        return (HashedDenseInstance) instance;
    }

    public static Instance hashInstanceAndDataset(Instance instance) {
        if (instance instanceof HashedDenseInstance) {
            hashInstances(instance.dataset());
            return instance;
        }
        Instances dataset = instance.dataset();
        int index = dataset.indexOf(instance);
        hashInstances(dataset);
        return dataset.get(index);
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        //TimeSeriesInstance are already hashable out of the box.
        //this is just so that it plays nicely with other code.
        return inst;
    }

    @Override
    public TimeSeriesInstances transform(TimeSeriesInstances data) {
        //TimeSeriesInstance are already hashable out of the box.
        //this is just so that it plays nicely with other code.
        return data;
    }

}
