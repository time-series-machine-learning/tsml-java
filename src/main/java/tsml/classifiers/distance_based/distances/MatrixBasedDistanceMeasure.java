package tsml.classifiers.distance_based.distances;

import java.util.Arrays;

/**
 * Abstract distance measure. This takes the weka interface for DistanceFunction and implements some default methods,
 * adding several checks and balances also. All distance measures should extends this class. This is loosely based on
 * the Transformer pattern whereby the user optionally "fits" some data and can then proceed to use the distance
 * measure. Simple distance measures need not fit at all, therefore the fit method is empty for those implementations
 * . fit() should always be called before any distance measurements.
 * <p>
 * Contributors: goastler
 */
public abstract class MatrixBasedDistanceMeasure extends BaseDistanceMeasure {

    private boolean recordCostMatrix = false;
    // the distance matrix produced by the distance function
    private double[][] costMatrix;
    private double[] oddRow;
    private double[] evenRow;
    private int numCols;
    private boolean recycleRows;

    /**
     * Indicate that a new distance is being computed and a corresponding matrix or pair or rows are required
     * @param numRows
     * @param numCols
     */
    protected void setup(int numRows, int numCols, boolean recycleRows) {
        oddRow = null;
        evenRow = null;
        costMatrix = null;
        this.numCols = numCols;
        this.recycleRows = recycleRows;
        if(recordCostMatrix) {
            costMatrix = new double[numRows][numCols];
            for(double[] array : costMatrix) Arrays.fill(array, getFillerValue());
        } else {
            oddRow = new double[numCols];
            evenRow = new double[numCols];
        }
    }
    
    protected double getFillerValue() {
        return Double.POSITIVE_INFINITY;
    }

    /**
     * Indicate that distance has been computed and any resources can be discarded. This preserves the distance matrix if set to do so, and discards all other resources. This is helpful to avoid the DistanceMeasure(s) retaining various rows / matrices post computation, never to be needed again but remaining in use in memory.
     */
    protected void teardown() {
        oddRow = null;
        evenRow = null;
        numCols = -1;
        recycleRows = false;
        if(!recordCostMatrix) {
            costMatrix = null;
        }
    }

    /**
     * Get a specified row. This manages the matrix automatically, returning the corresponding row, or recycles the rows if using a paired rows approach, or allocates a fresh row as required.
     * @param i
     * @return
     */
    protected double[] getRow(int i) {
        if(recordCostMatrix) {
            return costMatrix[i];
        } else if(recycleRows) {
            return i % 2 == 0 ? evenRow : oddRow;
        } else {
            return new double[numCols];
        }
    }
    
    public double[][] costMatrix() {
        return costMatrix;
    }

    public void clear() {
        costMatrix = null;
    }

    public boolean isRecordCostMatrix() {
        return recordCostMatrix;
    }

    public void setRecordCostMatrix(final boolean recordCostMatrix) {
        this.recordCostMatrix = recordCostMatrix;
    }
}
