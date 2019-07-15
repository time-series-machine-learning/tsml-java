package classifiers.distance_based.knn;

import distances.DistanceMeasure;
import distances.time_domain.dtw.Dtw;
import timeseriesweka.classifiers.CheckpointClassifier;
import utilities.ArrayUtilities;
import utilities.Copyable;
import utilities.IndividualOptionHandler;
import weka.core.Instance;
import weka.core.OptionHandler;

import java.util.Collection;
import java.util.List;

public class KnnConfiguration
    extends IndividualOptionHandler implements Copyable<KnnConfiguration> {
    // configuration options
    private final static String K_KEY = "k";
    private int k = 1;
    private final static String DISTANCE_MEASURE_KEY = "dm";
    private DistanceMeasure distanceMeasure = new Dtw(0);
    private final static String EARLY_ABANDON_KEY = "ea";
    private boolean earlyAbandon = false;
    // restriction options
    private final static String TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY = "trnsl";
    private int trainNeighbourhoodSizeLimit = -1;
    private final static String TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY = "trnslp";
    private double trainNeighbourhoodSizeLimitPercentage = -1;
    private final static String TRAIN_ESTIMATE_SET_SIZE_LIMIT = "tressl";
    private int trainEstimateSetSizeLimit = -1;
    private final static String TRAIN_ESTIMATE_SET_SIZE_LIMIT_PERCENTAGE = "tresslp";
    private double trainEstimateSetSizeLimitPercentage = -1;
    private final static String TEST_NEIGHBOURHOOD_SIZE_LIMIT = "tensl";
    private int testNeighbourhoodSizeLimit = -1;
    private final static String TEST_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE = "tenslp";
    private double testNeighbourhoodSizeLimitPercentage = -1;
    // iteration options
    private final static String TRAIN_NEIGHBOUR_SEARCH_STRATEGY_KEY = "trnss";
    private NeighbourSearchStrategy trainNeighbourSearchStrategy = NeighbourSearchStrategy.RANDOM;
    private final static String TRAIN_ESTIMATION_SOURCE_KEY = "trsss";
    private TrainEstimationSource trainEstimationSource = TrainEstimationSource.FROM_TRAIN_SET;
    private final static String TRAIN_ESTIMATION_STRATEGY_KEY = "tres";
    private TrainEstimationStrategy trainEstimationStrategy = TrainEstimationStrategy.RANDOM;
    // sets
    private List<Instance> predefinedTrainNeighbourhood = null;
    private List<Instance> predefinedTrainEstimateSet = null;

    public KnnConfiguration() {

    }

    public KnnConfiguration(KnnConfiguration other) throws
                                                    Exception {
        copyFrom(other);
    }

    public boolean mustResetTrain(KnnConfiguration other) {
        if(other.k < k) return true;
        else if(earlyAbandon != other.earlyAbandon) return true;
        else if(hasTrainEstimateSetSizeLimit() && trainEstimateSetSizeLimit < other.trainEstimateSetSizeLimit) return true;
        else if(hasTrainNeighbourhoodSizeLimit() && trainNeighbourhoodSizeLimit < other.trainNeighbourhoodSizeLimit) return true;
        else if(!trainNeighbourSearchStrategy.equals(other.trainNeighbourSearchStrategy)) return true;
        else if(!trainEstimationSource.equals(other.trainEstimationSource)) return true;
        else if(!trainEstimationStrategy.equals(other.trainEstimationStrategy)) return true;
        else if(!predefinedTrainEstimateSet.equals(other.predefinedTrainEstimateSet)) return true;
        else if(!predefinedTrainNeighbourhood.equals(other.predefinedTrainNeighbourhood)) return true;
        return false;
    }

    public boolean mustResetTest(KnnConfiguration other) {
        if(hasTestNeighbourhoodSizeLimit() && testNeighbourhoodSizeLimit < other.testNeighbourhoodSizeLimit) return true;
        return false;
    }

    public boolean hasTrainNeighbourhoodSizeLimit() {
        return trainNeighbourhoodSizeLimit >= 0;
    }

    public boolean hasTrainEstimateSetSizeLimit() {
        return trainEstimateSetSizeLimit >= 0;
    }

    public boolean hasTestNeighbourhoodSizeLimit() {
        return testNeighbourhoodSizeLimit >= 0;
    }

    @Override
    public String[] getOptions() {
        return ArrayUtilities.concat(distanceMeasure.getOptions(), new String[] {
            DISTANCE_MEASURE_KEY,
            distanceMeasure.toString(),
            K_KEY,
            String.valueOf(k),
            EARLY_ABANDON_KEY,
            String.valueOf(earlyAbandon),
            TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY,
            String.valueOf(trainNeighbourhoodSizeLimit),
            TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY,
            String.valueOf(trainNeighbourhoodSizeLimitPercentage),
            TRAIN_ESTIMATE_SET_SIZE_LIMIT,
            String.valueOf(trainEstimateSetSizeLimit),
            TRAIN_ESTIMATE_SET_SIZE_LIMIT_PERCENTAGE,
            String.valueOf(trainEstimateSetSizeLimitPercentage),
            TEST_NEIGHBOURHOOD_SIZE_LIMIT,
            String.valueOf(testNeighbourhoodSizeLimit),
            TEST_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE,
            String.valueOf(testNeighbourhoodSizeLimitPercentage),
            TRAIN_NEIGHBOUR_SEARCH_STRATEGY_KEY,
            String.valueOf(trainNeighbourSearchStrategy),
            TRAIN_ESTIMATION_SOURCE_KEY,
            String.valueOf(trainEstimationSource),
            TRAIN_ESTIMATION_STRATEGY_KEY,
            String.valueOf(trainEstimationStrategy),
            });
    }

    @Override
    public void setOption(String key, String value) {
        switch (key) {
            case K_KEY:
                setK(Integer.parseInt(value));
                break;
            case DISTANCE_MEASURE_KEY:
                setDistanceMeasure(DistanceMeasure.fromString(value));
                break;
            case EARLY_ABANDON_KEY:
                setEarlyAbandon(Boolean.parseBoolean(value));
                break;
            case TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY:
                setTrainNeighbourhoodSizeLimit(Integer.parseInt(value));
                break;
            case TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY:
                setTrainNeighbourhoodSizeLimitPercentage(Double.parseDouble(value));
                break;
            case TRAIN_ESTIMATE_SET_SIZE_LIMIT:
                setTrainEstimateSetSizeLimit(Integer.parseInt(value));
                break;
            case TRAIN_ESTIMATE_SET_SIZE_LIMIT_PERCENTAGE:
                setTrainEstimateSetSizeLimitPercentage(Double.parseDouble(value));
                break;
            case TEST_NEIGHBOURHOOD_SIZE_LIMIT:
                setTestNeighbourhoodSizeLimit(Integer.parseInt(value));
                break;
            case TEST_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE:
                setTestNeighbourhoodSizeLimitPercentage(Double.parseDouble(value));
                break;
            case TRAIN_NEIGHBOUR_SEARCH_STRATEGY_KEY:
                setTrainNeighbourSearchStrategy(NeighbourSearchStrategy.fromString(value));
                break;
            case TRAIN_ESTIMATION_SOURCE_KEY:
                setTrainEstimationSource(TrainEstimationSource.fromString(value));
                break;
            case TRAIN_ESTIMATION_STRATEGY_KEY:
                setTrainEstimationStrategy(TrainEstimationStrategy.fromString(value));
                break;
        }
    }

    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
    }

    public boolean isEarlyAbandon() {
        return earlyAbandon;
    }

    public void setEarlyAbandon(final boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
    }

    public int getTrainNeighbourhoodSizeLimit() {
        return trainNeighbourhoodSizeLimit;
    }

    public void setTrainNeighbourhoodSizeLimit(final int trainNeighbourhoodSizeLimit) {
        this.trainNeighbourhoodSizeLimit = trainNeighbourhoodSizeLimit;
    }

    public double getTrainNeighbourhoodSizeLimitPercentage() {
        return trainNeighbourhoodSizeLimitPercentage;
    }

    public void setTrainNeighbourhoodSizeLimitPercentage(final double trainNeighbourhoodSizeLimitPercentage) {
        this.trainNeighbourhoodSizeLimitPercentage = trainNeighbourhoodSizeLimitPercentage;
    }

    public int getTrainEstimateSetSizeLimit() {
        return trainEstimateSetSizeLimit;
    }

    public void setTrainEstimateSetSizeLimit(final int trainEstimateSetSizeLimit) {
        this.trainEstimateSetSizeLimit = trainEstimateSetSizeLimit;
    }

    public double getTrainEstimateSetSizeLimitPercentage() {
        return trainEstimateSetSizeLimitPercentage;
    }

    public void setTrainEstimateSetSizeLimitPercentage(final double trainEstimateSetSizeLimitPercentage) {
        this.trainEstimateSetSizeLimitPercentage = trainEstimateSetSizeLimitPercentage;
    }

    public int getTestNeighbourhoodSizeLimit() {
        return testNeighbourhoodSizeLimit;
    }

    public void setTestNeighbourhoodSizeLimit(final int testNeighbourhoodSizeLimit) {
        this.testNeighbourhoodSizeLimit = testNeighbourhoodSizeLimit;
    }

    public double getTestNeighbourhoodSizeLimitPercentage() {
        return testNeighbourhoodSizeLimitPercentage;
    }

    public void setTestNeighbourhoodSizeLimitPercentage(final double testNeighbourhoodSizeLimitPercentage) {
        this.testNeighbourhoodSizeLimitPercentage = testNeighbourhoodSizeLimitPercentage;
    }

    public NeighbourSearchStrategy getTrainNeighbourSearchStrategy() {
        return trainNeighbourSearchStrategy;
    }

    public void setTrainNeighbourSearchStrategy(final NeighbourSearchStrategy trainNeighbourSearchStrategy) {
        this.trainNeighbourSearchStrategy = trainNeighbourSearchStrategy;
    }

    public TrainEstimationSource getTrainEstimationSource() {
        return trainEstimationSource;
    }

    public void setTrainEstimationSource(final TrainEstimationSource trainEstimationSource) {
        this.trainEstimationSource = trainEstimationSource;
    }

    public TrainEstimationStrategy getTrainEstimationStrategy() {
        return trainEstimationStrategy;
    }

    public void setTrainEstimationStrategy(final TrainEstimationStrategy trainEstimationStrategy) {
        this.trainEstimationStrategy = trainEstimationStrategy;
    }

    public List<Instance> getPredefinedTrainNeighbourhood() {
        return predefinedTrainNeighbourhood;
    }

    public void setPredefinedTrainNeighbourhood(final List<Instance> predefinedTrainNeighbourhood) {
        this.predefinedTrainNeighbourhood = predefinedTrainNeighbourhood;
    }

    public List<Instance> getPredefinedTrainEstimateSet() {
        return predefinedTrainEstimateSet;
    }

    public void setPredefinedTrainEstimateSet(final List<Instance> predefinedTrainEstimateSet) {
        this.predefinedTrainEstimateSet = predefinedTrainEstimateSet;
    }

    public void setupNeighbourhoodSize(Collection<Instance> trainSet) {
        if (trainNeighbourhoodSizeLimitPercentage >= 0) {
            trainNeighbourhoodSizeLimit = (int) (trainSet.size() * trainNeighbourhoodSizeLimitPercentage);
        }
    }

    public void setupTrainEstimateSetSize(Collection<Instance> trainSet) {
        if (trainEstimateSetSizeLimitPercentage >= 0) {
            trainEstimateSetSizeLimit = (int) (trainSet.size() * trainEstimateSetSizeLimitPercentage);
        }
    }

    @Override
    public KnnConfiguration copy() throws
                                   Exception {
        KnnConfiguration configuration = new KnnConfiguration();
        configuration.copyFrom(this);
        return configuration;
    }

    @Override
    public void copyFrom(final Object object) throws
                                              Exception {
        KnnConfiguration other = (KnnConfiguration) object;
        setOptions(other.getOptions());
        predefinedTrainEstimateSet = other.predefinedTrainEstimateSet;
        predefinedTrainNeighbourhood = other.predefinedTrainNeighbourhood;
    }
}
