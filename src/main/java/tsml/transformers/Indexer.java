package tsml.transformers;

import akka.util.Index;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

import org.junit.Assert;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Objects;

public class Indexer extends BaseTrainableTransformer {

    public static Instances index(Instances data) {
        new Indexer().fit(data);
        return data;
    }

    public Indexer() {
        setHashInsteadOfIndex(false);
        reset();
    }

    private int index;
    private boolean hashInsteadOfIndex;

    @Override
    public void reset() {
        super.reset();
        index = -1;
    }

    public int size() {
        return index + 1;
    }

    @Override
    public void fit(final Instances data) {
        super.fit(data);
        boolean missingIndices = false;
        if (!hashInsteadOfIndex) {
            for (final Instance instance : data) {
                if (instance instanceof IndexedInstance) {
                    index = Math.max(index, ((IndexedInstance) instance).getIndex());
                } else {
                    missingIndices = true;
                }
            }
        }
        if (missingIndices) {
            for (int i = 0; i < data.size(); i++) {
                final Instance instance = data.get(i);
                int index;
                if (hashInsteadOfIndex) {
                    index = Arrays.hashCode(instance.toDoubleArray());
                } else {
                    index = ++this.index;
                }
                final IndexedInstance indexedInstance = new IndexedInstance(instance, index);
                data.set(i, indexedInstance);
            }
        }
    }

    @Override
    public Instance transform(final Instance inst) {
        return new IndexedInstance(inst, -1);
    }

    public Instance transform(final IndexedInstance inst) {
        return inst;
    }

    @Override
    public Instances determineOutputFormat(final Instances data) throws IllegalArgumentException {
        return new Instances(data, data.size());
    }

    public boolean isHashInsteadOfIndex() {
        return hashInsteadOfIndex;
    }

    public void setHashInsteadOfIndex(final boolean hashInsteadOfIndex) {
        this.hashInsteadOfIndex = hashInsteadOfIndex;
    }

    public static class IndexedInstance implements Instance, Serializable {

        public IndexedInstance(Instance instance, int index) {
            setIndex(index);
            setInstance(instance);
        }

        public IndexedInstance(IndexedInstance indexedInstance) {
            this((Instance) indexedInstance.instance.copy(), indexedInstance.index);
        }

        private int index;
        private Instance instance;

        public Instance getInstance() {
            return instance;
        }

        public void setInstance(final Instance instance) {
            Assert.assertNotNull(instance);
            this.instance = instance;
        }

        @Override
        public boolean equals(final Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof IndexedInstance)) {
                return false;
            }
            final IndexedInstance that = (IndexedInstance) o;
            return index == that.index;
        }

        @Override
        public int hashCode() {
            return index;
        }

        public int getIndex() {
            return index;
        }

        public void setIndex(final int index) {
            this.index = index;
        }

        @Override
        public Attribute attribute(final int index) {
            return instance.attribute(index);
        }

        @Override
        public Attribute attributeSparse(final int indexOfIndex) {
            return instance.attributeSparse(indexOfIndex);
        }

        @Override
        public Attribute classAttribute() {
            return instance.classAttribute();
        }

        @Override
        public int classIndex() {
            return instance.classIndex();
        }

        @Override
        public boolean classIsMissing() {
            return instance.classIsMissing();
        }

        @Override
        public double classValue() {
            return instance.classValue();
        }

        @Override
        public Instances dataset() {
            return instance.dataset();
        }

        @Override
        public void deleteAttributeAt(final int position) {
            instance.deleteAttributeAt(position);
        }

        @Override
        public Enumeration enumerateAttributes() {
            return instance.enumerateAttributes();
        }

        @Override
        public boolean equalHeaders(final Instance inst) {
            return instance.equalHeaders(inst);
        }

        @Override
        public String equalHeadersMsg(final Instance inst) {
            return instance.equalHeadersMsg(inst);
        }

        @Override
        public boolean hasMissingValue() {
            return instance.hasMissingValue();
        }

        @Override
        public int index(final int position) {
            return instance.index(position);
        }

        @Override
        public void insertAttributeAt(final int position) {
            instance.insertAttributeAt(position);
        }

        @Override
        public boolean isMissing(final int attIndex) {
            return instance.isMissing(attIndex);
        }

        @Override
        public boolean isMissingSparse(final int indexOfIndex) {
            return instance.isMissingSparse(indexOfIndex);
        }

        @Override
        public boolean isMissing(final Attribute att) {
            return instance.isMissing(att);
        }

        @Override
        public Instance mergeInstance(final Instance inst) {
            return instance.mergeInstance(inst);
        }

        @Override
        public int numAttributes() {
            return instance.numAttributes();
        }

        @Override
        public int numClasses() {
            return instance.numClasses();
        }

        @Override
        public int numValues() {
            return instance.numValues();
        }

        @Override
        public void replaceMissingValues(final double[] array) {
            instance.replaceMissingValues(array);
        }

        @Override
        public void setClassMissing() {
            instance.setClassMissing();
        }

        @Override
        public void setClassValue(final double value) {
            instance.setClassValue(value);
        }

        @Override
        public void setClassValue(final String value) {
            instance.setClassValue(value);
        }

        @Override
        public void setDataset(final Instances instances) {
            instance.setDataset(instances);
        }

        @Override
        public void setMissing(final int attIndex) {
            instance.setMissing(attIndex);
        }

        @Override
        public void setMissing(final Attribute att) {
            instance.setMissing(att);
        }

        @Override
        public void setValue(final int attIndex, final double value) {
            instance.setValue(attIndex, value);
        }

        @Override
        public void setValueSparse(final int indexOfIndex, final double value) {
            instance.setValueSparse(indexOfIndex, value);
        }

        @Override
        public void setValue(final int attIndex, final String value) {
            instance.setValue(attIndex, value);
        }

        @Override
        public void setValue(final Attribute att, final double value) {
            instance.setValue(att, value);
        }

        @Override
        public void setValue(final Attribute att, final String value) {
            instance.setValue(att, value);
        }

        @Override
        public void setWeight(final double weight) {
            instance.setWeight(weight);
        }

        @Override
        public Instances relationalValue(final int attIndex) {
            return instance.relationalValue(attIndex);
        }

        @Override
        public Instances relationalValue(final Attribute att) {
            return instance.relationalValue(att);
        }

        @Override
        public String stringValue(final int attIndex) {
            return instance.stringValue(attIndex);
        }

        @Override
        public String stringValue(final Attribute att) {
            return instance.stringValue(att);
        }

        @Override
        public double[] toDoubleArray() {
            return instance.toDoubleArray();
        }

        @Override
        public String toStringNoWeight(final int afterDecimalPoint) {
            return instance.toStringNoWeight(afterDecimalPoint);
        }

        @Override
        public String toStringNoWeight() {
            return instance.toStringNoWeight();
        }

        @Override
        public String toStringMaxDecimalDigits(final int afterDecimalPoint) {
            return instance.toStringMaxDecimalDigits(afterDecimalPoint);
        }

        @Override
        public String toString(final int attIndex, final int afterDecimalPoint) {
            return instance.toString(attIndex, afterDecimalPoint);
        }

        @Override
        public String toString(final int attIndex) {
            return instance.toString(attIndex);
        }

        @Override
        public String toString(final Attribute att, final int afterDecimalPoint) {
            return instance.toString(att, afterDecimalPoint);
        }

        @Override
        public String toString(final Attribute att) {
            return instance.toString(att);
        }

        @Override
        public double value(final int attIndex) {
            return instance.value(attIndex);
        }

        @Override
        public double valueSparse(final int indexOfIndex) {
            return instance.valueSparse(indexOfIndex);
        }

        @Override
        public double value(final Attribute att) {
            return instance.value(att);
        }

        @Override
        public double weight() {
            return instance.weight();
        }

        @Override
        public IndexedInstance copy() {
            return new IndexedInstance(this);
        }

        @Override
        public String toString() {
            return "IndexedInstance{" + "index=" + index + ", instance=" + instance + '}';
        }
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        // TODO Auto-generated method stub

    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        // TODO Auto-generated method stub
        return null;
    }
}
