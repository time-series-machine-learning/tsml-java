package tsml.classifiers.distance_based.distances;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.PerformanceStats;

import java.util.Objects;

/**
 * Abstract distance measure. This takes the weka interface for DistanceFunction and implements some default methods,
 * adding several checks and balances also. All distance measures should extends this class. This is loosely based on
 * the Transformer pattern whereby the user optionally "fits" some data and can then proceed to use the distance
 * measure. Simple distance measures need not fit at all, therefore the fit method is empty for those implementations
 * . fit() should always be called before any distance measurements.
 * <p>
 * Contributors: goastler
 */
public abstract class MatrixBasedDistanceMeasure implements DistanceMeasure {

    private boolean generateDistanceMatrix = false;
    // the distance matrix produced by the distance function
    private double[][] matrix;

    public double[][] getDistanceMatrix() {
        return matrix;
    }

    protected void setDistanceMatrix(double[][] matrix) {
        if(isGenerateDistanceMatrix()) {
            this.matrix = matrix;
        }
    }

    public void cleanDistanceMatrix() {
        matrix = null;
    }

    public boolean isGenerateDistanceMatrix() {
        return generateDistanceMatrix;
    }

    public void setGenerateDistanceMatrix(final boolean generateDistanceMatrix) {
        this.generateDistanceMatrix = generateDistanceMatrix;
    }
    
    protected void checkData(TimeSeriesInstance a, TimeSeriesInstance b, double limit) {
        Objects.requireNonNull(a);
        Objects.requireNonNull(b);
        if(a.getNumDimensions() != b.getNumDimensions()) {
            throw new IllegalArgumentException("differing number of dimensions in the two instances");
        }
        if(limit < 0) {
            throw new IllegalArgumentException("limit less than zero");
        }
    }

}
