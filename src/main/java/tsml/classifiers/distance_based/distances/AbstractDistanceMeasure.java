package tsml.classifiers.distance_based.distances;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

import java.io.Serializable;
import java.util.Enumeration;

public abstract class AbstractDistanceMeasure
    implements DistanceMeasure {

    protected transient boolean debug = false;
    protected transient boolean dataHasBeenSet = false;
    protected transient Instances data;

    @Override
    public boolean isDebug() {
        return debug;
    }

    @Override
    public void setDebug(final boolean debug) {
        this.debug = debug;
    }

    public AbstractDistanceMeasure() { }

    public void checks(Instance first, Instance second) {
        if(first.classIndex() != first.numAttributes() - 1) {
            throw new IllegalStateException("class value must be at the end");
        }
        if(second.classIndex() != second.numAttributes() - 1) {
            throw new IllegalStateException("class value must be at the end");
        }
        if(!dataHasBeenSet) {
            throw new IllegalStateException("must call setInstances first to setup the distance measure");
        }
    }

    @Override
    public void setInstances(final Instances data) {
        this.data = data;
        dataHasBeenSet = true;
    }

    @Override
    public Instances getInstances() {
        return data;
    }

    @Override
    public void setAttributeIndices(final String value) {

    } // todo

    @Override
    public String getAttributeIndices() {
        return null;
    } // todo

    @Override
    public void setInvertSelection(final boolean value) {

    }

    @Override
    public boolean getInvertSelection() {
        return false;
    }

    @Override
    public void postProcessDistances(final double[] distances) {

    }

    @Override
    public void update(final Instance ins) {
        data.add(ins);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    public boolean isSymmetric() {
        return true;
    }

//    public int hashcode() {
//        return toString().hashCode();
//    }
//
//    public boolean equals(Object other) {
//        if(other instanceof AbstractDistanceMeasure) {
//            return hashcode() == other.hashCode();
//        } else {
//            return false;
//        }
//    }

}
