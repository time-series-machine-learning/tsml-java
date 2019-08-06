package timeseriesweka.classifiers;

public interface Seedable {
    void setTrainSeed(long seed);
    void setTestSeed(long seed);
    Long getTrainSeed();
    Long getTestSeed();
}
