package tsml.transformers;

import java.util.Objects;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: transforms a set of instances into a set of hashed instances. This allows them to reliably be cached.
 * <p>
 * Contributors: goastler, abostrom
 */

public class Indexer implements Transformer {

    @Override
    public Instances determineOutputFormat(final Instances inputFormat) {
        return new Instances(inputFormat, 0);
    }

    @Override
    public Instances transform(Instances inst) {
        if(!inst.isEmpty()) {
            transform(inst.get(0), true);
        }
        return inst;
    }

    @Override
    public Instance transform(Instance inst) {
        if(inst instanceof IndexedInstance) {
            return inst;
        }
        return transform(inst, false);
    }

    public Instance transform(Instance target, boolean transformAll) {
        Instances instances = target.dataset();
        IndexedInstance result = null;
        for(int i = 0; i < instances.size(); i++) {
            Instance inst = instances.get(i);
            final boolean found = inst == target;
            if(transformAll || found) {
                if(inst instanceof IndexedInstance) {
                    ((IndexedInstance) inst).setIndex(i);
                } else {
                    inst = new IndexedInstance(inst, i);
                    instances.set(i, inst);
                }
                if(found) {
                    result = (IndexedInstance) inst;
                }
            }
        }
        return result;
    }

    public static class IndexedInstance extends DenseInstance {
        private int index;

        public IndexedInstance(final Instance instance, final int index) {
            super(instance);
            setIndex(index);
        }

        public int getIndex() {
            return index;
        }

        private void setIndex(final int index) {
            this.index = index;
        }

        @Override
        public double[] toDoubleArray() {
            // directly exposes double[] to outsiders
            return m_AttValues;
        }

        @Override
        public boolean equals(final Object o) {
            if(this == o) {
                return true;
            }
            if(!(o instanceof IndexedInstance)) {
                return false;
            }
            final IndexedInstance that = (IndexedInstance) o;
            return index == that.index;
        }

        @Override
        public int hashCode() {
            return index;
        }
    }

    public static double[] extractAttributeValuesAndClassLabel(Instance instance) {
        if(!(instance instanceof IndexedInstance)) {
            instance = new IndexedInstance(instance, -1);
        }
        return instance.toDoubleArray();
    }
}
