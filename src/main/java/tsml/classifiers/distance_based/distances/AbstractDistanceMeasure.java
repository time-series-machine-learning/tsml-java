package tsml.classifiers.distance_based.distances;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Abstract distance measure. This takes the weka interface for DistanceFunction and implements some default methods,
 * adding several checks and balances also. All distance measures should extends this class.
 *
 * Contributors: goastler
 */
public abstract class AbstractDistanceMeasure
    implements DistanceMeasure {
    // simple debug switch
    private transient boolean debug = false;
    // check for whether setInstances has been called before doing any distance measurements
    private transient boolean dataAvailable = false;
    // the data which was passed to setInstances
    private transient Instances data;

    @Override
    public boolean isDebug() {
        return debug;
    }

    @Override
    public void setDebug(final boolean debug) {
        this.debug = debug;
    }

    public AbstractDistanceMeasure() {}

    // optional check for data in the correct format
    public void checkData(Instance first, Instance second) {
        if(!dataAvailable) {
            throw new IllegalStateException("must call setInstances first to setup the distance measure");
        }
    }

    @Override
    public void setInstances(final Instances data) {
        this.data = data;
        dataAvailable = data != null;
        if(dataAvailable) {
            if(data.classIndex() != data.numAttributes() - 1) {
                throw new IllegalStateException("class value must be at the end");
            }
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

    // whether the distance measure is symmetric (i.e. dist from inst A to inst B == dist from inst B to inst A
    public boolean isSymmetric() {
        return true;
    }

    public void clean() {
        data = null;
        dataAvailable = false;
    }
}
