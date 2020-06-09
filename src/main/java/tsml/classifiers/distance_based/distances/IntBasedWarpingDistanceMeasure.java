package tsml.classifiers.distance_based.distances;

public abstract class IntBasedWarpingDistanceMeasure extends WarpingDistanceMeasure {

    // the distance matrix produced by the distance function
    protected int[][] matrix;

    public int[][] getMatrix() {
        return matrix;
    }

    protected void setMatrix(int[][] matrix) {
        if(keepMatrix) {
            this.matrix = matrix;
        }
    }

    public void cleanDistanceMatrix() {
        matrix = null;
    }
}
