package tsml.transformers;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

/**
 * Purpose: transforms a set of instances into a set of hashed instances. This allows them to reliably be cached.
 * <p>
 * Contributors: goastler, abostrom
 */

public class Indexer extends BaseTrainableTransformer {

    private int index;

    public Indexer() {
        reset();
    }

    public void reset() {
        super.reset();
        index = -1;
    }

    @Override
    public Instances determineOutputFormat(final Instances inputFormat) {
        return new Instances(inputFormat, 0);
    }

    public int size() {
        return index + 1;
    }

    @Override
    public void fit(final Instances data) {
        super.fit(data);
        // find max index
        for(int i = 0; i < data.size(); i++) {
            final Instance instance = data.get(i);
            if(instance instanceof IndexedInstance) {
                index = Math.max(index, ((IndexedInstance) instance).getIndex());
            }
        }
        // loop through un-indexed instances and index them
        for(int i = 0; i < data.size(); i++) {
            Instance inst = data.get(i);
            if(!(inst instanceof IndexedInstance)) {
                index++;
                inst = new IndexedInstance(inst, index);
                inst.setDataset(data);
                data.set(i, inst);
            }
        }
    }

    @Override
    public IndexedInstance transform(Instance inst) {
        if(!isFit()) {
            throw new IllegalStateException("must be fitted first");
        }
        if(inst instanceof IndexedInstance) {
            return (IndexedInstance) inst;
        }
        return new IndexedInstance(inst, -1);
    }

    public static class IndexedInstance extends DenseInstance {
        private int index;

        public IndexedInstance(final Instance instance, final int index) {
            super(instance);
            setIndex(index);
            setDataset(instance.dataset());
        }

        public IndexedInstance(IndexedInstance instance) {
            this(instance, instance.getIndex());
        }

        public IndexedInstance(double weight, double[] attValues, int index) {
            super(weight, attValues);
            setIndex(index);
        }

        public int getIndex() {
            return index;
        }

        private void setIndex(final int index) {
            this.index = index;
        }

        @Override public String toString() {
            return "IndexedInstance{" +
                   "index=" + index +
                   ", label=" + classValue() +
                   '}';
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

        @Override
        public IndexedInstance copy() {
            return new IndexedInstance(this);
        }
    }

    public static double[] extractAttributeValuesAndClassLabel(Instance instance) {
        if(!(instance instanceof IndexedInstance)) {
            instance = new IndexedInstance(instance, -1);
        }
        return instance.toDoubleArray();
    }

    public static List<IndexedInstance> index(Instances instances) {
        new Indexer().fit(instances);
        List<IndexedInstance> list = new ArrayList<>();
        for(Instance instance : instances) {
            list.add((IndexedInstance) instance);
        }
        return list;
    }

}
