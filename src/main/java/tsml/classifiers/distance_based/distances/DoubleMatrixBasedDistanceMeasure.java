package tsml.classifiers.distance_based.distances;

public abstract class DoubleMatrixBasedDistanceMeasure extends MatrixBasedDistanceMeasure {

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
}
