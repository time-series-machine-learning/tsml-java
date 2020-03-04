package tsml.filters;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

import java.util.Arrays;
import java.util.List;

public class HashFilter extends SimpleBatchFilter {

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected Instances determineOutputFormat(final Instances inputFormat) throws
                                                                           Exception {
        return inputFormat;
    }

    @Override public boolean setInputFormat(final Instances instanceInfo) throws Exception {
        hashInstances(instanceInfo);
        return super.setInputFormat(instanceInfo);
    }

    @Override
    public Instances process(final Instances instances) throws
                                                           Exception {
        hashInstances(instances); // this can be removed when this func is made protected as it should be
        return instances;
    }

    public static class HashedDenseInstance
        extends DenseInstance {
        private int id;

        public HashedDenseInstance(Instance instance) {
            super(instance);
            setDataset(instance.dataset());
            if(instance instanceof HashedDenseInstance) {
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
            if(o instanceof HashedDenseInstance) {
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
        for(int i = 0; i < instances.size(); i++) {
            Instance instance = instances.get(i);
            if(!(instance instanceof HashedDenseInstance)) {
                HashedDenseInstance indexedInstance = hashInstance(instance);
                instances.set(i, indexedInstance);
            }
        }
    }

    public static HashedDenseInstance hashInstance(Instance instance) {
        if(!(instance instanceof HashedDenseInstance)) {
            instance = new HashedDenseInstance(instance);
        }
        return (HashedDenseInstance) instance;
    }

    public static Instance hashInstanceAndDataset(Instance instance) {
        if(instance instanceof HashedDenseInstance) {
            hashInstances(instance.dataset());
            return instance;
        }
        Instances dataset = instance.dataset();
        int index = dataset.indexOf(instance);
        hashInstances(dataset);
        return dataset.get(index);
    }
}
