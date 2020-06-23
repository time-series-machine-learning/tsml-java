package tsml.classifiers.distance_based.utils.intervals;

import org.junit.Assert;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

public class IntervalInstance implements Instance {
    private Interval interval;
    private Instance instance;

    public IntervalInstance(final Interval interval, final Instance instance) {
        setInstance(instance);
        setInterval(interval);
    }

    public Interval getInterval() {
        return interval;
    }

    public void setInterval(final Interval interval) {
        Assert.assertNotNull(interval);
        this.interval = interval;
    }

    public Instance getInstance() {
        return instance;
    }

    public void setInstance(final Instance instance) {
        Assert.assertNotNull(instance);
        this.instance = instance;
    }

    @Override public Attribute attribute(int index) {
        index = interval.translate(index);
        return instance.attribute(index);
    }

    @Override public Attribute attributeSparse(int indexOfIndex) {
        indexOfIndex = interval.translate(indexOfIndex);
        return instance.attributeSparse(indexOfIndex);
    }

    @Override public Attribute classAttribute() {
        return instance.classAttribute();
    }

    @Override public int classIndex() {
        return interval.size();
    }

    @Override public boolean classIsMissing() {
        return instance.classIsMissing();
    }

    @Override public double classValue() {
        return instance.classValue();
    }

    @Override public Instances dataset() {
        return instance.dataset();
    }

    @Override public void deleteAttributeAt(int position) {
        position = interval.translate(position);
        instance.deleteAttributeAt(position);
    }

    @Override public Enumeration enumerateAttributes() {
        return instance.enumerateAttributes();
    }

    @Override public boolean equalHeaders(final Instance inst) {
        return instance.equalHeaders(inst);
    }

    @Override public String equalHeadersMsg(final Instance inst) {
        return instance.equalHeadersMsg(inst);
    }

    @Override public boolean hasMissingValue() {
        return instance.hasMissingValue();
    }

    @Override public int index(int position) {
        position = interval.translate(position);
        return instance.index(position);
    }

    @Override public void insertAttributeAt(int position) {
        position = interval.translate(position);
        instance.insertAttributeAt(position);
    }

    @Override public boolean isMissing(int attIndex) {
        attIndex = interval.translate(attIndex);
        return instance.isMissing(attIndex);
    }

    @Override public boolean isMissingSparse(int indexOfIndex) {
        indexOfIndex = interval.translate(indexOfIndex);
        return instance.isMissingSparse(indexOfIndex);
    }

    @Override public boolean isMissing(final Attribute att) {
        return instance.isMissing(att);
    }

    @Override public Instance mergeInstance(final Instance inst) {
        return instance.mergeInstance(inst);
    }

    @Override public int numAttributes() {
        return interval.size() + 1; // +1 for class value
    }

    @Override public int numClasses() {
        return instance.numClasses();
    }

    @Override public int numValues() {
        return instance.numValues();
    }

    @Override public void replaceMissingValues(final double[] array) {
        instance.replaceMissingValues(array);
    }

    @Override public void setClassMissing() {
        instance.setClassMissing();
    }

    @Override public void setClassValue(final double value) {
        instance.setClassValue(value);
    }

    @Override public void setClassValue(final String value) {
        instance.setClassValue(value);
    }

    @Override public void setDataset(final Instances instances) {
        instance.setDataset(instances);
    }

    @Override public void setMissing(final int attIndex) {
        instance.setMissing(attIndex);
    }

    @Override public void setMissing(final Attribute att) {
        instance.setMissing(att);
    }

    @Override public void setValue(int attIndex, final double value) {
        attIndex = interval.translate(attIndex);
        instance.setValue(attIndex, value);
    }

    @Override public void setValueSparse(int indexOfIndex, final double value) {
        indexOfIndex = interval.translate(indexOfIndex);
        instance.setValueSparse(indexOfIndex, value);
    }

    @Override public void setValue(int attIndex, final String value) {
        attIndex = interval.translate(attIndex);
        instance.setValue(attIndex, value);
    }

    @Override public void setValue(final Attribute att, final double value) {
        instance.setValue(att, value);
    }

    @Override public void setValue(final Attribute att, final String value) {
        instance.setValue(att, value);
    }

    @Override public void setWeight(final double weight) {
        instance.setWeight(weight);
    }

    @Override public Instances relationalValue(int attIndex) {
        attIndex = interval.translate(attIndex);
        return instance.relationalValue(attIndex);
    }

    @Override public Instances relationalValue(final Attribute att) {
        return instance.relationalValue(att);
    }

    @Override public String stringValue(int attIndex) {
        attIndex = interval.translate(attIndex);
        return instance.stringValue(attIndex);
    }

    @Override public String stringValue(final Attribute att) {
        return instance.stringValue(att);
    }

    @Override public double[] toDoubleArray() {
        return instance.toDoubleArray();
    }

    @Override public String toStringNoWeight(final int afterDecimalPoint) {
        return instance.toStringNoWeight(afterDecimalPoint);
    }

    @Override public String toStringNoWeight() {
        return instance.toStringNoWeight();
    }

    @Override public String toStringMaxDecimalDigits(final int afterDecimalPoint) {
        return instance.toStringMaxDecimalDigits(afterDecimalPoint);
    }

    @Override public String toString(int attIndex, final int afterDecimalPoint) {
        attIndex = interval.translate(attIndex);
        return instance.toString(attIndex, afterDecimalPoint);
    }

    @Override public String toString(int attIndex) {
        attIndex = interval.translate(attIndex);
        return instance.toString(attIndex);
    }

    @Override public String toString(final Attribute att, final int afterDecimalPoint) {
        return instance.toString(att, afterDecimalPoint);
    }

    @Override public String toString(final Attribute att) {
        return instance.toString(att);
    }

    @Override public double value(int attIndex) {
        attIndex = interval.translate(attIndex);
        return instance.value(attIndex);
    }

    @Override public double valueSparse(int indexOfIndex) {
        indexOfIndex = interval.translate(indexOfIndex);
        return instance.valueSparse(indexOfIndex);
    }

    @Override public double value(final Attribute att) {
        return instance.value(att);
    }

    @Override public double weight() {
        return instance.weight();
    }

    @Override public IntervalInstance copy() {
        return new IntervalInstance(interval, instance);
    }
}
