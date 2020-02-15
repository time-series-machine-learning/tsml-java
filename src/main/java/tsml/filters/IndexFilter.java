package tsml.filters;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

import java.util.Arrays;
import java.util.List;

public class IndexFilter extends SimpleBatchFilter {

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected Instances determineOutputFormat(final Instances inputFormat) throws
                                                                           Exception {
        return inputFormat;
    }

    @Override
    public Instances process(final Instances instances) throws
                                                           Exception {
        hashifyInstances(instances);
        return instances;
    }

    public static class HashableDenseInstance
        extends DenseInstance {
        private int hashcode;
        private boolean buildHashcode = true;

        public HashableDenseInstance(HashableDenseInstance instance) {
            this((Instance) instance);
            hashcode = instance.hashcode;
            buildHashcode = instance.buildHashcode;
        }

        public HashableDenseInstance(Instance instance) {
            super(instance);
            setDataset(instance.dataset());
        }

        @Override
        public int hashCode() {
            if(buildHashcode) {
                hashcode = Arrays.hashCode(m_AttValues);
            }
            return hashcode;
        }

        @Override
        public boolean equals(final Object o) {
            if(!(o instanceof HashableDenseInstance)) {
                return false;
            }
            HashableDenseInstance other = (HashableDenseInstance) o;
            return Arrays.equals(m_AttValues, other.m_AttValues);
        }

        @Override
        public Object copy() {
            return new HashableDenseInstance(this);
        }
    }

    public static void hashifyInstances(List<Instance> instances) {
        for(int i = 0; i < instances.size(); i++) {
            Instance instance = instances.get(i);
            if(!(instance instanceof HashableDenseInstance)) {
                HashableDenseInstance indexedInstance = hashifyInstance(instance);
                instances.set(i, indexedInstance);
            }
        }
    }

    public static HashableDenseInstance hashifyInstance(Instance instance) {
        if(!(instance instanceof HashableDenseInstance)) {
            instance = new HashableDenseInstance(instance);
        }
        return (HashableDenseInstance) instance;
    }

    public static Instance hashifyInstanceAndDataset(Instance instance) {
        if(instance instanceof HashableDenseInstance) {
            hashifyInstances(instance.dataset());
            return instance;
        }
        Instances dataset = instance.dataset();
        int index = dataset.indexOf(instance);
        hashifyInstances(dataset);
        return dataset.get(index);
    }
}
