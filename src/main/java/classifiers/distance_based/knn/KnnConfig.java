package classifiers.distance_based.knn;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.knn.sampling.DistributedRandomSampler;
import classifiers.distance_based.knn.sampling.LinearSampler;
import classifiers.distance_based.knn.sampling.RandomSampler;
import classifiers.distance_based.knn.sampling.RoundRobinRandomSampler;
import classifiers.template.configuration.TemplateConfig;
import distances.DistanceMeasure;
import distances.time_domain.dtw.Dtw;
import utilities.ArrayUtilities;
import weka.core.Instance;

import java.util.Collection;
import java.util.List;
import java.util.Random;

public class KnnConfig
    extends TemplateConfig {
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
    private final static String TRAIN_NEIGHBOURHOOD_SIZE_THRESHOLD = "trnst";
    private int trainNeighbourhoodSizeThreshold = -1;
    private final static String TRAIN_NEIGHBOURHOOD_SIZE_THRESHOLD_PERCENTAGE = "trnstp";
    private double trainNeighbourhoodSizeThresholdPercentage = -1;
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

    public static class TrainIterators {
        private final DynamicIterator<Instance, ?> trainSetIterator;
        private final DynamicIterator<Instance, ?> trainEstimateSetIterator;

        public TrainIterators(final DynamicIterator<Instance, ?> trainSetIterator,
                              final DynamicIterator<Instance, ?> trainEstimateSetIterator) {
            this.trainSetIterator = trainSetIterator;
            this.trainEstimateSetIterator = trainEstimateSetIterator;
        }

        public DynamicIterator<Instance, ?> getTrainEstimateSetIterator() {
            return trainEstimateSetIterator;
        }

        public DynamicIterator<Instance, ?> getTrainSetIterator() {
            return trainSetIterator;
        }
    }

    public TrainIterators buildTrainIterators(Collection<Instance> trainSet, Random random) {
        DynamicIterator<Instance, ?> trainEstimateSetIterator = buildTrainEstimationStrategy(trainSet, random);
        DynamicIterator<Instance, ?> trainSetIterator = buildNeighbourSearchStrategy(trainSet, random);
        if(trainEstimationSource.equals(TrainEstimationSource.FROM_TRAIN_NEIGHBOURHOOD)) {
            final DynamicIterator<Instance, ?> finalTrainSetIterator = trainSetIterator;
            trainSetIterator = new DynamicIterator<Instance, DynamicIterator<Instance, ?>>() {

                private final DynamicIterator<Instance, ?> iterator = finalTrainSetIterator;

                @Override
                public DynamicIterator<Instance, ?> iterator() {
                    return null; // todo hmmmmm
                }

                @Override
                public boolean hasNext() {
                    return iterator.hasNext();
                }

                @Override
                public Instance next() {
                    Instance next = iterator.next();
                    trainEstimateSetIterator.add(next);
                    return next;
                }

                @Override
                public void remove() {
                    iterator.remove();
                }

                @Override
                public void add(final Instance instance) {
                    iterator.add(instance);
                }
            };
        }
        return new TrainIterators(trainSetIterator, trainEstimateSetIterator);
    }

    public DynamicIterator<Instance, ?> buildNeighbourSearchStrategy(Collection<Instance> trainSet, Random random) {
        DynamicIterator<Instance, ?> iterator;
        switch (getTrainNeighbourSearchStrategy()) {
            case RANDOM:
                iterator = new RandomSampler(random);
                break;
            case LINEAR:
                iterator = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                iterator = new RoundRobinRandomSampler(random);
                break;
            case DISTRIBUTED_RANDOM:
                iterator = new DistributedRandomSampler(random);
                break;
            default:
                throw new UnsupportedOperationException();
        }
        if(getPredefinedTrainNeighbourhood() != null) {
            iterator.addAll(getPredefinedTrainNeighbourhood());
        } else {
            iterator.addAll(trainSet);
        }
        return iterator;
    }

    public DynamicIterator<Instance, ?> buildTrainEstimationStrategy(Collection<Instance> trainSet, Random random) {
        DynamicIterator<Instance, ?> iterator;
        switch (getTrainEstimationStrategy()) {
            case RANDOM:
                iterator = new RandomSampler(random);
                break;
            case LINEAR:
                iterator = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                iterator = new RoundRobinRandomSampler(random);
                break;
            case DISTRIBUTED_RANDOM:
                iterator = new DistributedRandomSampler(random);
                break;
            default:
                throw new UnsupportedOperationException();
        }
        if(getPredefinedTrainEstimateSet() != null) {
            iterator.addAll(getPredefinedTrainEstimateSet());
        } else {
            switch (getTrainEstimationSource()) {
                case FROM_TRAIN_SET:
                    iterator.addAll(trainSet);
                    break;
                case FROM_TRAIN_NEIGHBOURHOOD:
                    // add the train neighbours as sampled from train set
                    break;
                default:
                    throw new IllegalStateException("train estimation source unknown");
            }
        }
        return iterator;
    }

    public KnnConfig() {

    }

    public KnnConfig(KnnConfig other) throws
                                      Exception {
        super(other);
    }


    public boolean mustResetTrain(KnnConfig other) {
        if(other.k < k) return true;
        else if(earlyAbandon != other.earlyAbandon) return true;
        else if(hasTrainEstimateSetSizeLimit() && trainEstimateSetSizeLimit < other.trainEstimateSetSizeLimit) return true;
        else if(hasTrainNeighbourhoodSizeLimit() && trainNeighbourhoodSizeLimit < other.trainNeighbourhoodSizeLimit) return true;
        else if(!trainNeighbourSearchStrategy.equals(other.trainNeighbourSearchStrategy)) return true;
        else if(!trainEstimationSource.equals(other.trainEstimationSource)) return true;
        else if(!trainEstimationStrategy.equals(other.trainEstimationStrategy)) return true;
        else if(!predefinedTrainEstimateSet.equals(other.predefinedTrainEstimateSet)) return true;
        else if (!predefinedTrainNeighbourhood.equals(other.predefinedTrainNeighbourhood)) return true;
        else {
            return hasTrainNeighbourhoodSizeThreshold() && trainNeighbourhoodSizeThreshold != other.trainNeighbourhoodSizeThreshold;
        }
    }

    public boolean mustResetTest(KnnConfig other) {
        return hasTestNeighbourhoodSizeLimit() && testNeighbourhoodSizeLimit < other.testNeighbourhoodSizeLimit;
    }

    public boolean hasTrainNeighbourhoodSizeThreshold() {
        return trainNeighbourhoodSizeThreshold >= 0;
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
            TRAIN_NEIGHBOURHOOD_SIZE_THRESHOLD,
            String.valueOf(trainNeighbourhoodSizeThreshold),
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
            case TRAIN_NEIGHBOURHOOD_SIZE_THRESHOLD:
                setTrainNeighbourhoodSizeThreshold(Integer.parseInt(value));
                break;
            case TRAIN_NEIGHBOURHOOD_SIZE_THRESHOLD_PERCENTAGE:
                setTrainNeighbourhoodSizeThresholdPercentage(Double.parseDouble(value));
        }
    }

    public int getTrainNeighbourhoodSizeThreshold() {
        return trainNeighbourhoodSizeThreshold;
    }

    public void setTrainNeighbourhoodSizeThreshold(final int trainNeighbourhoodSizeThreshold) {
        this.trainNeighbourhoodSizeThreshold = trainNeighbourhoodSizeThreshold;
    }

    public double getTrainNeighbourhoodSizeThresholdPercentage() {
        return trainNeighbourhoodSizeThresholdPercentage;
    }

    public void setTrainNeighbourhoodSizeThresholdPercentage(final double trainNeighbourhoodSizeThresholdPercentage) {
        this.trainNeighbourhoodSizeThresholdPercentage = trainNeighbourhoodSizeThresholdPercentage;
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

    public void setupTrainNeighbourhoodSizeThreshold(Collection<Instance> trainSet) {
        if(trainNeighbourhoodSizeThresholdPercentage >= 0) {
            trainNeighbourhoodSizeThreshold = (int) (trainSet.size() * trainNeighbourhoodSizeThresholdPercentage);
        }
    }

    @Override
    public KnnConfig copy() throws
                                   Exception {
        KnnConfig configuration = new KnnConfig();
        configuration.copyFrom(this);
        return configuration;
    }

    @Override
    public void copyFrom(final Object object) throws
                                              Exception {
        KnnConfig other = (KnnConfig) object;
        setOptions(other.getOptions());
        predefinedTrainEstimateSet = other.predefinedTrainEstimateSet;
        predefinedTrainNeighbourhood = other.predefinedTrainNeighbourhood;
    }
}
