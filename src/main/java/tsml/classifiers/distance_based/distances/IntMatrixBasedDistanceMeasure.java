package tsml.classifiers.distance_based.distances;

public abstract class IntMatrixBasedDistanceMeasure extends MatrixBasedDistanceMeasure {

    // the distance matrix produced by the distance function
    private int[][] matrix;

    public int[][] getDistanceMatrix() {
        return matrix;
    }

    protected void setDistanceMatrix(int[][] matrix) {
        if(isGenerateDistanceMatrix()) {
            this.matrix = matrix;
        }
    }

    public void cleanDistanceMatrix() {
        matrix = null;
    }
}
