package tsml.classifiers.distance_based.distances;

import weka.core.Instance;
import weka.core.Instances;

public abstract class AbstractDistanceMeasure
    implements DistanceMeasure {

    private transient boolean debug = false;
    private transient boolean dataHasBeenSet = false;
    private transient Instances data;

    @Override
    public boolean isDebug() {
        return debug;
    }

    @Override
    public void setDebug(final boolean debug) {
        this.debug = debug;
    }

    public AbstractDistanceMeasure() { }

    public void checkData(Instance first, Instance second) {
        if(!dataHasBeenSet) {
            throw new IllegalStateException("must call setInstances first to setup the distance measure");
        }
    }

    @Override
    public void setInstances(final Instances data) {
        this.data = data;
        dataHasBeenSet = true;
        if(data.classIndex() != data.numAttributes() - 1) {
            throw new IllegalStateException("class value must be at the end");
        }
    }

    @Override
    public Instances getInstances() {
        return data;
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

}
