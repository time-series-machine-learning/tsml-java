package tsml.classifiers.distance_based.distances;

public abstract class DoubleBasedWarpingDistanceMeasure extends WarpingDistanceMeasure {

    // the distance matrix produced by the distance function
    protected double[][] matrix;

    public double[][] getMatrix() {
        return matrix;
    }

    protected void setMatrix(double[][] matrix) {
        if(keepMatrix) {
            this.matrix = matrix;
        }
    }

    public void cleanDistanceMatrix() {
        matrix = null;
    }
}
