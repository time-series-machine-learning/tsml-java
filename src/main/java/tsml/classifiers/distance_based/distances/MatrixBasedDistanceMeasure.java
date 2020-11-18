package tsml.classifiers.distance_based.distances;

public abstract class MatrixBasedDistanceMeasure extends BaseDistanceMeasure {

    private boolean generateDistanceMatrix = false;

    public boolean isGenerateDistanceMatrix() {
        return generateDistanceMatrix;
    }

    public void setGenerateDistanceMatrix(final boolean generateDistanceMatrix) {
        this.generateDistanceMatrix = generateDistanceMatrix;
    }

    public abstract void cleanDistanceMatrix();
}
