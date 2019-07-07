package classifiers.distance_based.knn;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.knn.sampling.*;
import classifiers.template_classifier.TemplateClassifier;
import distances.DistanceMeasure;
import distances.time_domain.dtw.Dtw;
import evaluation.storage.ClassifierResults;
import utilities.ArrayUtilities;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class Knn
    extends TemplateClassifier<Knn> {

    public int getK() {
        return k;
    }

    public void setK(int k) {
        if(k < 1) {
            throw new IllegalArgumentException("k cannot be less than 1");
        }
        this.k = k;
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(DistanceMeasure distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
    }

    public boolean isEarlyAbandon() {
        return earlyAbandon;
    }

    public void setEarlyAbandon(boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
    }

    public int getTrainNeighbourhoodSizeLimit() {
        return trainNeighbourhoodSizeLimit;
    }

    public void setTrainNeighbourhoodSizeLimit(int trainNeighbourhoodSizeLimit) {
        this.trainNeighbourhoodSizeLimit = trainNeighbourhoodSizeLimit;
    }

    public double getTrainNeighbourhoodSizeLimitPercentage() {
        return trainNeighbourhoodSizeLimitPercentage;
    }

    public void setTrainNeighbourhoodSizeLimitPercentage(double trainNeighbourhoodSizeLimitPercentage) {
        this.trainNeighbourhoodSizeLimitPercentage = trainNeighbourhoodSizeLimitPercentage;
    }

    public int getTrainEstimateSetSizeLimit() {
        return trainEstimateSetSizeLimit;
    }

    public void setTrainEstimateSetSizeLimit(int trainEstimateSetSizeLimit) {
        this.trainEstimateSetSizeLimit = trainEstimateSetSizeLimit;
    }

    public double getTrainEstimateSetSizeLimitPercentage() {
        return trainEstimateSetSizeLimitPercentage;
    }

    public void setTrainEstimateSetSizeLimitPercentage(double trainEstimateSetSizeLimitPercentage) {
        this.trainEstimateSetSizeLimitPercentage = trainEstimateSetSizeLimitPercentage;
    }

    public NeighbourSearchStrategy getTrainNeighbourSearchStrategy() {
        return trainNeighbourSearchStrategy;
    }

    public void setTrainNeighbourSearchStrategy(NeighbourSearchStrategy trainNeighbourSearchStrategy) {
        this.trainNeighbourSearchStrategy = trainNeighbourSearchStrategy;
    }

    public TrainEstimationSource getTrainEstimationSource() {
        return trainEstimationSource;
    }

    public void setTrainEstimationSource(TrainEstimationSource trainEstimationSource) {
        this.trainEstimationSource = trainEstimationSource;
    }

    public TrainEstimationStrategy getTrainEstimationStrategy() {
        return trainEstimationStrategy;
    }

    public void setTrainEstimationStrategy(TrainEstimationStrategy trainEstimationStrategy) {
        this.trainEstimationStrategy = trainEstimationStrategy;
    }

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
    private List<KNearestNeighbours> trainEstimate = null;
    private List<Instance> trainSet = null;
    private List<Instance> trainNeighbourhood = null;
    private List<Instance> predefinedTrainNeighbourhood = null;
    private List<Instance> predefinedTrainEstimateSet = null;
    // iterators for executing strategies
    private DynamicIterator<Instance, ?> trainNeighbourIterator = null;
    private DynamicIterator<Instance, ?> trainEstimatorIterator = null;

    public Knn() {}

    public Knn(Knn other) throws
                          Exception {
        super(other);
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

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("haven't done this yet!");
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

    public double getTestNeighbourhoodSizeLimitPercentage() {
        return testNeighbourhoodSizeLimitPercentage;
    }
    public void setTestNeighbourhoodSizeLimitPercentage(double testNeighbourhoodSizeLimitPercentage) {
        this.testNeighbourhoodSizeLimitPercentage = testNeighbourhoodSizeLimitPercentage;
    }

    public int getTestNeighbourhoodSizeLimit() {
        return testNeighbourhoodSizeLimit;
    }
    public void setTestNeighbourhoodSizeLimit(int testNeighbourhoodSizeLimit) {
        this.testNeighbourhoodSizeLimit = testNeighbourhoodSizeLimit;
    }
    
    @Override
    public String toString() {
        return "KNN";
    }

    private void setupTrainSet(Instances trainSet) {
        if (trainSetChanged(trainSet)) { // todo call in setters if exceeding certain
            // stuff, e.g. trainNeighbourhood size inc
            getTrainStopWatch().reset();
            this.trainSet = trainSet;
            trainEstimate = new ArrayList<>();
            trainNeighbourhood = new ArrayList<>();
            setupNeighbourhoodSize();
            setupTrainEstimateSetSize();
            setupNeighbourSearchStrategy();
            setupTrainEstimationStrategy();
        }
    }

    private void setupNeighbourSearchStrategy() {
        switch (trainNeighbourSearchStrategy) {
            case RANDOM:
                trainNeighbourIterator = new RandomSampler(getTrainRandom().nextLong());
                break;
            case LINEAR:
                trainNeighbourIterator = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                trainNeighbourIterator = new RoundRobinRandomSampler(getTrainRandom());
                break;
            case DISTRIBUTED_RANDOM:
                trainNeighbourIterator = new DistributedRandomSampler(getTrainRandom());
                break;
            default:
                throw new UnsupportedOperationException();
        }
        if(predefinedTrainNeighbourhood != null) {
            trainNeighbourIterator.addAll(predefinedTrainNeighbourhood);
        } else {
            trainNeighbourIterator.addAll(trainSet);
        }
    }

    private void setupNeighbourhoodSize() {
        if (trainNeighbourhoodSizeLimitPercentage >= 0) {
            trainNeighbourhoodSizeLimit = (int) (trainSet.size() * trainNeighbourhoodSizeLimitPercentage);
        }
    }

    private void setupTrainEstimateSetSize() {
        if (trainEstimateSetSizeLimitPercentage >= 0) {
            trainEstimateSetSizeLimit = (int) (trainSet.size() * trainEstimateSetSizeLimitPercentage);
        }
    }

    private void setupTrainEstimationStrategy() {
        switch (trainEstimationStrategy) {
            case RANDOM:
                trainEstimatorIterator = new RandomSampler(getTrainRandom().nextLong()); // todo make the remainder of these use seed instead of random
                break;
            case LINEAR:
                trainEstimatorIterator = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                trainEstimatorIterator = new RoundRobinRandomSampler(getTrainRandom());
                break;
            case DISTRIBUTED_RANDOM:
                trainEstimatorIterator = new DistributedRandomSampler(getTrainRandom());
                break;
            default:
                throw new UnsupportedOperationException();
        }
        if(predefinedTrainEstimateSet != null) {
            trainEstimatorIterator.addAll(predefinedTrainEstimateSet);
        } else {
            switch (trainEstimationSource) {
                case FROM_TRAIN_SET:
                    trainEstimatorIterator.addAll(trainSet);
                    break;
                case FROM_TRAIN_NEIGHBOURHOOD:
                    // add the train neighbours as sampled from train set
                    break;
                default:
                    throw new IllegalStateException("train estimation source unknown");
            }
        }
    }


    private boolean hasRemainingTrainEstimations() {
        return (
                    trainEstimate.size() < trainEstimateSetSizeLimit // if train estimate set under limit
                    || trainEstimateSetSizeLimit < 0 // if train estimate limit set
               )
               && trainEstimatorIterator.hasNext(); // if remaining train estimators
    }

    private boolean hasRemainingNeighbours() {
        return (
                    trainNeighbourhood.size() < trainNeighbourhoodSizeLimit // if within train neighbourhood size limit
                    || trainNeighbourhoodSizeLimit < 0 // if active train neighbourhood size limit
               )
               && trainNeighbourIterator.hasNext(); // if there are remaining neighbours
    }

    private void nextTrainEstimator() {
        Instance instance = trainEstimatorIterator.next();
        trainEstimatorIterator.remove();
        KNearestNeighbours kNearestNeighbours = new KNearestNeighbours(instance);
        trainEstimate.add(kNearestNeighbours);
        kNearestNeighbours.addAll(trainNeighbourhood);
    }

    private void nextNeighbourSearch() {
        Instance trainNeighbour = trainNeighbourIterator.next();
        trainNeighbourIterator.remove();
        trainNeighbourhood.add(trainNeighbour);
        for (KNearestNeighbours trainEstimator : this.trainEstimate) {
            trainEstimator.add(trainNeighbour);
        }
        if(trainEstimationSource == TrainEstimationSource.FROM_TRAIN_NEIGHBOURHOOD) {
            trainEstimatorIterator.add(trainNeighbour);
        }
    }

    private void buildTrainResults() throws Exception {
        ClassifierResults trainResults = new ClassifierResults();
        setTrainResults(trainResults);
        for (KNearestNeighbours KNearestNeighbours : trainEstimate) {
            KNearestNeighbours.trim();
            Prediction prediction = KNearestNeighbours.predict();
            double[] distribution = prediction.getDistribution();
            trainResults.addPrediction(KNearestNeighbours.getTarget().classValue(),
                    distribution,
                    ArrayUtilities.indexOfMax(distribution, getTrainRandom()),
                    prediction.getPredictionTimeNanos(),
                    null);
        }
        setClassifierResultsMetaInfo(trainResults);
    }

    @Override
    public void buildClassifier(Instances trainSet) throws
                                                           Exception {
        setupTrainSet(trainSet);
        boolean remainingTrainEstimations = hasRemainingTrainEstimations();
        boolean remainingNeighbours = hasRemainingNeighbours();
        boolean choice = true;
        while ((
                    remainingTrainEstimations
                    || remainingNeighbours
                )
                && withinTrainContract()) {
            if(remainingTrainEstimations && remainingNeighbours) {
                choice = !choice;//getTrainRandom().nextBoolean(); // todo change to strategy
            } else choice = !remainingNeighbours;
            if(choice) {
//            if(remainingTrainEstimations) {
                nextTrainEstimator();
            } else {
                nextNeighbourSearch();
            }
            remainingTrainEstimations = hasRemainingTrainEstimations();
            remainingNeighbours = hasRemainingNeighbours();
            getTrainStopWatch().lap();
        }
        buildTrainResults();
        getTrainStopWatch().lap();
    }

    public Knn copy() { // todo copyable interface
        Knn knn = new Knn();
        try {
            knn.copyFrom(this);
            return knn;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void copyFrom(final Object object) throws
                                                    Exception {
        super.copyFromSerObject(object);
        Knn other = (Knn) object;
        trainEstimate.clear();
        for (KNearestNeighbours KNearestNeighbours : other.trainEstimate) {
            trainEstimate.add(new KNearestNeighbours(KNearestNeighbours));
        }
        setOptions(other.getOptions());
        trainNeighbourhood.clear();
        trainNeighbourhood.addAll(other.trainNeighbourhood);
        trainEstimatorIterator = other.trainEstimatorIterator.iterator();
        trainNeighbourIterator = other.trainNeighbourIterator.iterator();
        trainSet = other.trainSet;
        predefinedTrainEstimateSet = other.predefinedTrainEstimateSet;
        predefinedTrainNeighbourhood = other.predefinedTrainNeighbourhood;
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                         Exception {
        KNearestNeighbours testKNearestNeighbours = new KNearestNeighbours(testInstance);
        testKNearestNeighbours.addAll(trainSet); // todo limited test neighbourhood
        testKNearestNeighbours.trim(); // todo empty train set should be rand predict
        return testKNearestNeighbours.predict().getDistribution();
    }

    public enum NeighbourSearchStrategy {
        RANDOM,
        LINEAR,
        ROUND_ROBIN_RANDOM,
        DISTRIBUTED_RANDOM;

        public static NeighbourSearchStrategy fromString(String str) {
            for (NeighbourSearchStrategy s : NeighbourSearchStrategy.values()) {
                if (s.name()
                     .equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    public enum TrainEstimationStrategy {
        RANDOM,
        LINEAR,
        ROUND_ROBIN_RANDOM,
        DISTRIBUTED_RANDOM;

        public static TrainEstimationStrategy fromString(String str) {
            for (TrainEstimationStrategy s : values()) {
                if (s.name()
                        .equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    public enum TrainEstimationSource {
        FROM_TRAIN_NEIGHBOURHOOD,
        FROM_TRAIN_SET;

        public static TrainEstimationSource fromString(String str) {
            for (TrainEstimationSource s : values()) {
                if (s.name()
                     .equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    private static class Prediction { // todo use time units
        private final double[] distribution;
        private final long predictionTimeNanos;

        private Prediction(double[] distribution, long predictionTimeNanos) {
            this.distribution = distribution;
            this.predictionTimeNanos = predictionTimeNanos;
        }

        public double[] getDistribution() {
            return distribution;
        }

        public long getPredictionTimeNanos() {
            return predictionTimeNanos;
        }

    }

    private class KNearestNeighbours {

        private final Instance target;
        private final TreeMap<Double, Collection<Instance>> kNeighbours = new TreeMap<>();
        private TreeMap<Double, Collection<Instance>> trimmedKNeighbours = kNeighbours;
        private int size = 0;
        private long searchTimeNanos = 0;
        private Collection<Instance> furthestNeighbours = null;
        private double furthestDistance = Double.POSITIVE_INFINITY;

        private KNearestNeighbours(final Instance target) {
            this.target = target;
        }

        public KNearestNeighbours(KNearestNeighbours other) {
            this(other.target);
            for (Map.Entry<Double, Collection<Instance>> entry : other.kNeighbours.entrySet()) {
                kNeighbours.put(entry.getKey(), new ArrayList<>(entry.getValue()));
            }
            size = other.size;
            searchTimeNanos = other.searchTimeNanos;
            furthestDistance = kNeighbours.lastKey();
            furthestNeighbours = kNeighbours.lastEntry().getValue();
            trim();
        }

        public void trim() {
            long startTime = System.nanoTime();
            if(size <= k) {
                trimmedKNeighbours = kNeighbours;
            } else {
                trimmedKNeighbours = new TreeMap<>();
                List<Instance> furthestNeighbours = null;
                for (Map.Entry<Double, Collection<Instance>> entry : kNeighbours.entrySet()) {
                    furthestNeighbours = new ArrayList<>(entry.getValue());
                    trimmedKNeighbours.put(entry.getKey(), furthestNeighbours);
                }
                if(furthestNeighbours != null) {
                    int size = furthestNeighbours.size();
                    while (size > k) {
                        size--;
                        int index = getTrainRandom().nextInt(furthestNeighbours.size());
                        furthestNeighbours.remove(index);
                    }
                }
            }
            searchTimeNanos = System.nanoTime() - startTime;
        }

        public Instance getTarget() {
            return target;
        }

        public int size() {
            return size;
        }

        public Prediction predict() {
            long startTime = System.nanoTime();
            double[] distribution = new double[target.numClasses()];
            TreeMap<Double, Collection<Instance>> neighbours = trimmedKNeighbours;
            if (neighbours.size() == 0) {
                distribution[getTestRandom().nextInt(distribution.length)]++;
            } else {
                for (Map.Entry<Double, Collection<Instance>> entry : neighbours.entrySet()) {
                    for (Instance instance : entry.getValue()) {
                        // todo weighted
                        distribution[(int) instance.classValue()]++;
                    }
                }
                ArrayUtilities.normaliseInplace(distribution);
            }
            long predictTimeNanos = System.nanoTime() - startTime;
            return new Prediction(distribution, predictTimeNanos);
        }

        public void addAll(final List<Instance> instances) {
            for (Instance instance : instances) {
                add(instance);
            }
        }

        public double add(Instance instance) {
            long startTime = System.nanoTime();
            double maxDistance = earlyAbandon ? furthestDistance : Double.POSITIVE_INFINITY;
            double distance = distanceMeasure.distance(target, instance, maxDistance);
            searchTimeNanos += System.nanoTime() - startTime;
            add(instance, distance);
            return distance;
        }

        public void add(Instance instance, double distance) {
            long startTime = System.nanoTime();
            if(!instance.equals(target)) {
                if((distance <= furthestDistance || size < k) && k > 0) {
                    Collection<Instance> equalDistanceNeighbours = kNeighbours.get(distance);
                    if (equalDistanceNeighbours == null) {
                        equalDistanceNeighbours = new ArrayList<>();
                        kNeighbours.put(distance, equalDistanceNeighbours);
                        if(size == 0) {
                            furthestDistance = distance;
                            furthestNeighbours = equalDistanceNeighbours;
                        } else {
                            furthestDistance = Math.max(furthestDistance, distance);
                        }
                    }
                    equalDistanceNeighbours.add(instance);
                    size++;
                    if(distance < furthestDistance && size > k) { // if we've got too many neighbours AND just added a neighbour closer than the furthest then try and knock off the furthest lot
                        int numFurthestNeighbours = furthestNeighbours.size();
                        if (size - k >= numFurthestNeighbours) {
                            kNeighbours.pollLastEntry();
                            size -= numFurthestNeighbours;
                            Map.Entry<Double, Collection<Instance>> furthestNeighboursEntry = kNeighbours.lastEntry();
                            furthestNeighbours = furthestNeighboursEntry.getValue();
                            furthestDistance = furthestNeighboursEntry.getKey();
                        }
                    }
                    trimmedKNeighbours = null;
                }
            }
            searchTimeNanos += System.nanoTime() - startTime;
        }
    }
}
