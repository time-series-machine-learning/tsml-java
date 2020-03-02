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
        private static int hashCodeGlobalIndex = 0;
        private int hashcode;
        private boolean buildHashcode = true;

        public HashedDenseInstance(HashedDenseInstance instance) {
            this((Instance) instance);
            hashcode = instance.hashcode;
            buildHashcode = instance.buildHashcode;
        }

        public HashedDenseInstance(Instance instance) {
            super(instance);
            setDataset(instance.dataset());
            hashcode = hashCodeGlobalIndex++;
        }

        @Override
        public int hashCode() {
//            if(buildHashcode) {
//                buildHashcode = false;
//                hashcode = Arrays.hashCode(m_AttValues);
//            }
            return hashcode;
        }

        @Override
        public boolean equals(final Object o) {
            if(!(o instanceof HashedDenseInstance)) {
                return false;
            }
            HashedDenseInstance other = (HashedDenseInstance) o;
            return Arrays.equals(m_AttValues, other.m_AttValues);
        }

        @Override
        public Object copy() {
            return new HashedDenseInstance(this);
        }

//        @Override public Instance mergeInstance(final Instance inst) {
//            buildHashcode = true;
//            return super.mergeInstance(inst);
//        }
//
//        @Override protected void forceDeleteAttributeAt(final int position) {
//            buildHashcode = true;
//            super.forceDeleteAttributeAt(position);
//        }
//
//        @Override protected void forceInsertAttributeAt(final int position) {
//            buildHashcode = true;
//            super.forceInsertAttributeAt(position);
//        }
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
