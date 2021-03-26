package tsml.classifiers.distance_based.knn;

public class Neighbour {
    public Neighbour(final double distance, final int indexInTrainData, final boolean wasNearestNeighbour) {
        this.distance = distance;
        this.indexInTrainData = indexInTrainData;
        this.wasNearestNeighbour = wasNearestNeighbour;
    }

    private final double distance;
    private final int indexInTrainData;
    private final boolean wasNearestNeighbour;

    public double getDistance() {
        return distance;
    }

    public int getIndexInTrainData() {
        return indexInTrainData;
    }

    /**
     * Is the neighbour a nearest neighbour? Note this is only valid at the time of adding the neighbour to the search.
     * I.e. it may no longer be a nearest neighbour if more neighbours have been examined in the search.
     * @return
     */
    public boolean wasNearestNeighbour() {
        return wasNearestNeighbour;
    }
}
