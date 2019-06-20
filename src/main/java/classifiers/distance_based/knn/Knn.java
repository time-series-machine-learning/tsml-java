package classifiers.distance_based.knn;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.knn.sampling.*;
import classifiers.template_classifier.TemplateClassifier;
import distances.DistanceMeasure;
import distances.time_domain.dtw.Dtw;
import evaluation.storage.ClassifierResults;
import utilities.ArrayUtilities;
import utilities.StringUtilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class Knn
    extends TemplateClassifier {

    public static final int DEFAULT_K = 1;
    public static final boolean DEFAULT_EARLY_ABANDON = true;
    public static final String K_KEY = "k";
    public static final int DEFAULT_TRAIN_NEIGHBOURHOOD_SIZE = -1;
    public static final double DEFAULT_TRAIN_NEIGHBOURHOOD_SIZE_PERCENTAGE = -1;
    public static final int DEFAULT_TEST_NEIGHBOURHOOD_SIZE = -1;
    public static final double DEFAULT_TEST_NEIGHBOURHOOD_SIZE_PERCENTAGE = -1;
    public static final String TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY = "trainNeighbourhoodSizeLimit";
    public static final String TEST_NEIGHBOURHOOD_SIZE_LIMIT_KEY = "testNeighbourhoodSizeLimit";
    public static final String TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY = "trainNeighbourhoodSizeLimitPercentage";
    public static final String TEST_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY = "testNeighbourhoodSizeLimitPercentage";
    public static final String DISTANCE_MEASURE_KEY = "distanceMeasure";
    public static final String NEIGHBOUR_SEARCH_STRATEGY_KEY = "neighbourSearchStrategy";
    public static final String TRAIN_ESTIMATE_SET_SIZE_KEY = "trainEstimateSetSizeLimit";
    public static final String TRAIN_ESTIMATE_SET_SIZE_PERCENTAGE_KEY = "trainEstimateSetSizeLimitPercentage";
    public static final String EARLY_ABANDON_KEY = "earlyAbandon";
    public static final int DEFAULT_TRAIN_ESTIMATE_SET_SIZE = -1;
    public static final int DEFAULT_TRAIN_ESTIMATE_SET_SIZE_PERCENTAGE = -1;
    public static final double DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE = -1;
    private final List<KNearestNeighbours> trainKNearestNeighbours = new ArrayList<>();
    private final List<Instance> trainEstimateSet = new ArrayList<>();
    private Instances trainSet;
    private int k;
    private DistanceMeasure distanceMeasure;
    private boolean earlyAbandon;
    private int trainNeighbourhoodSizeLimit;
    private double trainEstimateSetSizeLimitPercentage;
    private int trainEstimateSetSizeLimit;
    private double testNeighbourhoodSizeLimitPercentage;
    private int testNeighbourhoodSizeLimit;
    private NeighbourSearchStrategy neighbourSearchStrategy = NeighbourSearchStrategy.RANDOM;
    private DynamicIterator<Instance, ?> neighboursIterator;
    private DynamicIterator<Instance, ?> trainSetIterator;
    private List<Instance> trainNeighbourhood = new ArrayList<>();
    private List<Instance> predefinedTrainNeighbourhood = null; // todo predefined trainestimateset
    private double trainNeighbourhoodSizeLimitPercentage = DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE;
    private TrainSetSampleStrategy trainSetSampleStrategy = TrainSetSampleStrategy.RANDOM;

    public Knn() {
        setDistanceMeasure(new Dtw(0));
        setK(DEFAULT_K);
        setTrainNeighbourhoodSizeLimit(DEFAULT_TRAIN_NEIGHBOURHOOD_SIZE);
        setTrainNeighbourhoodSizeLimitPercentage(DEFAULT_TRAIN_NEIGHBOURHOOD_SIZE_PERCENTAGE);
        setEarlyAbandon(DEFAULT_EARLY_ABANDON);
        setTrainEstimateSetSizeLimitPercentage(DEFAULT_TRAIN_ESTIMATE_SET_SIZE_PERCENTAGE);
        setTrainEstimateSetSizeLimit(DEFAULT_TRAIN_ESTIMATE_SET_SIZE);
        setNeighbourSearchStrategy(NeighbourSearchStrategy.RANDOM);
        setTestNeighbourhoodSizeLimit(DEFAULT_TEST_NEIGHBOURHOOD_SIZE);
        setTestNeighbourhoodSizeLimitPercentage(DEFAULT_TEST_NEIGHBOURHOOD_SIZE_PERCENTAGE);
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

    public List<Instance> getPredefinedTrainNeighbourhood() {
        return predefinedTrainNeighbourhood;
    }

    public void setPredefinedTrainNeighbourhood(List<Instance> predefinedTrainNeighbourhood) {
        this.predefinedTrainNeighbourhood = new ArrayList<>(predefinedTrainNeighbourhood);
    }

    public TrainSetSampleStrategy getTrainSetSampleStrategy() {
        return trainSetSampleStrategy;
    }

    public void setTrainSetSampleStrategy(TrainSetSampleStrategy trainSetSampleStrategy) {
        this.trainSetSampleStrategy = trainSetSampleStrategy;
    }

    @Override
    public String toString() {
        return "knn";
    }

    public Knn copy() {
        Knn knn = new Knn();
        try {
            knn.copyFromSerObject(this);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return knn;
    }

    public int getK() {
        return k;
    }

    public void setK(final int k) {
        this.k = k;
    }

    public String[] getOptions() { // todo
        return ArrayUtilities.concat(super.getOptions(), new String[] {
            K_KEY,
            String.valueOf(k),
                TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY,
            String.valueOf(trainNeighbourhoodSizeLimit),
            NEIGHBOUR_SEARCH_STRATEGY_KEY,
            neighbourSearchStrategy.name(),
                TRAIN_ESTIMATE_SET_SIZE_KEY,
            String.valueOf(trainEstimateSetSizeLimit),
            EARLY_ABANDON_KEY,
            String.valueOf(earlyAbandon),
                TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY,
            String.valueOf(trainNeighbourhoodSizeLimitPercentage),
                TRAIN_ESTIMATE_SET_SIZE_PERCENTAGE_KEY,
            String.valueOf(trainEstimateSetSizeLimitPercentage),
            DISTANCE_MEASURE_KEY,
            distanceMeasure.toString(),
            StringUtilities.join(",", distanceMeasure.getOptions()),
            });
    }

    public void setOptions(final String[] options) throws
                                                   Exception { // todo
        super.setOptions(options);
        for (int i = 0; i < options.length - 1; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            if (key.equals(K_KEY)) {
                setK(Integer.parseInt(value));
            } else if (key.equals(TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY)) {
                setK(Integer.parseInt(value));
            } else if (key.equals(DISTANCE_MEASURE_KEY)) {
                setDistanceMeasure(DistanceMeasure.fromString(value));
            } else if (key.equals(NEIGHBOUR_SEARCH_STRATEGY_KEY)) {
                setNeighbourSearchStrategy(NeighbourSearchStrategy.fromString(value));
            } else if (key.equals(EARLY_ABANDON_KEY)) {
                setEarlyAbandon(Boolean.parseBoolean(value));
            } else if (key.equals(TRAIN_ESTIMATE_SET_SIZE_KEY)) {
                setTrainEstimateSetSizeLimit(Integer.parseInt(value));
            } else if (key.equals(TRAIN_ESTIMATE_SET_SIZE_PERCENTAGE_KEY)) {
                setTrainEstimateSetSizeLimitPercentage(Double.parseDouble(value));
            } else if (key.equals(TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY)) {
                setTrainNeighbourhoodSizeLimitPercentage(Double.valueOf(value));
            } else if(key.equals(TEST_NEIGHBOURHOOD_SIZE_LIMIT_KEY)) {
                setTestNeighbourhoodSizeLimit(Integer.parseInt(value));
            } else if(key.equals(TEST_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY)) {
                setTestNeighbourhoodSizeLimitPercentage(Double.parseDouble(value));
            }
        }
        distanceMeasure.setOptions(options);
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
    }

    public boolean getEarlyAbandon() {
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

    public NeighbourSearchStrategy getNeighbourSearchStrategy() {
        return neighbourSearchStrategy;
    }

    public void setNeighbourSearchStrategy(final NeighbourSearchStrategy neighbourSearchStrategy) {
        this.neighbourSearchStrategy = neighbourSearchStrategy;
    }

    public double getTrainEstimateSetSizeLimitPercentage() {
        return trainEstimateSetSizeLimitPercentage;
    }

    public void setTrainEstimateSetSizeLimitPercentage(final double trainEstimateSetSizeLimitPercentage) {
        this.trainEstimateSetSizeLimitPercentage = trainEstimateSetSizeLimitPercentage;
    }

    public int getTrainEstimateSetSizeLimit() {
        return trainEstimateSetSizeLimit;
    }

    public void setTrainEstimateSetSizeLimit(final int trainEstimateSetSizeLimit) {
        this.trainEstimateSetSizeLimit = trainEstimateSetSizeLimit;
    }

    public double getTrainNeighbourhoodSizeLimitPercentage() {
        return trainNeighbourhoodSizeLimitPercentage;
    }

    public void setTrainNeighbourhoodSizeLimitPercentage(double trainNeighbourhoodSizeLimitPercentage) {
        this.trainNeighbourhoodSizeLimitPercentage = trainNeighbourhoodSizeLimitPercentage;
    }

    private boolean hasRemainingTrainSubSet() {
        return (trainEstimateSet.size() < trainEstimateSetSizeLimit || (trainEstimateSetSizeLimit < 0 && trainEstimateSet.size() < trainSet.size()))
               && trainSetIterator.hasNext();
    }

    private boolean hasRemainingNeighbours() {
        return (trainNeighbourhood.size() < trainNeighbourhoodSizeLimit || trainNeighbourhoodSizeLimit < 0)
               && neighboursIterator.hasNext();
    }

    private Instance nextTrainInstance() {
        Instance instance = trainSetIterator.next();
        trainSetIterator.remove();
        return instance;
    }

    private void nextTrainEstimateInstance() {
        Instance trainInstance = nextTrainInstance();
        KNearestNeighbours kNearestNeighbours = new KNearestNeighbours(trainInstance);
        trainKNearestNeighbours.add(kNearestNeighbours);
        trainEstimateSet.add(trainInstance);
        kNearestNeighbours.addAll(trainNeighbourhood);
        if(predefinedTrainNeighbourhood == null) {
            neighboursIterator.add(trainInstance);
        }
    }

    private Instance nextNeighbour() {
        Instance neighbour = neighboursIterator.next();
        neighboursIterator.remove();
        return neighbour;
    }

    private void nextNeighbourSearch() {
        Instance neighbour = nextNeighbour();
        trainNeighbourhood.add(neighbour);
        for (KNearestNeighbours trainKNearestNeighbours : this.trainKNearestNeighbours) {
            trainKNearestNeighbours.add(neighbour);
        }
    }

    private void buildTrainResults() throws Exception {
        ClassifierResults trainResults = new ClassifierResults();
        setTrainResults(trainResults);
        for (KNearestNeighbours KNearestNeighbours : trainKNearestNeighbours) {
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

    private void checkTrainSet(Instances trainSet) {
        if (trainSetChanged(trainSet)) { // todo call in setters if exceeding certain
            // stuff, e.g. trainNeighbourhood size inc
            getTrainStopWatch().reset();
            this.trainSet = trainSet;
            trainKNearestNeighbours.clear();
            trainEstimateSet.clear();
            setupNeighbourhoodSize();
            setupTrainSetSize();
            setupTrainSetSampleStrategy();
            setupNeighbourSearchStrategy();
        }
    }

    @Override
    public void buildClassifier(Instances trainSet) throws
                                                           Exception {
        checkTrainSet(trainSet);
        boolean remainingTrainSubSet = hasRemainingTrainSubSet();
        boolean remainingNeighbours = hasRemainingNeighbours();
        while ((remainingTrainSubSet || remainingNeighbours)) {
//            boolean choice = true;
//            if(remainingTrainSubSet && remainingNeighbours) {
//                choice = getTrainRandom().nextBoolean(); // todo change to strategy
//            } else if(remainingNeighbours) { // todo empty train set should be rand predict
//                choice = false;
//            }
//            if(choice) {
            if(remainingTrainSubSet) {
                nextTrainEstimateInstance();
            } else {
                nextNeighbourSearch();
            }
            remainingTrainSubSet = hasRemainingTrainSubSet();
            remainingNeighbours = hasRemainingNeighbours();
            getTrainStopWatch().lap();
        }
        buildTrainResults();
        getTrainStopWatch().lap();
    }

    private void setupNeighbourSearchStrategy() {
        switch (neighbourSearchStrategy) {
            case RANDOM:
                neighboursIterator = new RandomSampler(getTrainRandom().nextLong());
                break;
            case LINEAR:
                neighboursIterator = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                neighboursIterator = new RoundRobinRandomSampler(getTrainRandom());
                break;
            case DISTRIBUTED_RANDOM:
                neighboursIterator = new DistributedRandomSampler(getTrainRandom());
                break;
            default:
                throw new UnsupportedOperationException();
        }
        if(predefinedTrainNeighbourhood != null) {
            neighboursIterator.addAll(predefinedTrainNeighbourhood);
        }
    }

    private void setupNeighbourhoodSize() {
        if (trainNeighbourhoodSizeLimitPercentage >= 0) {
            setTrainNeighbourhoodSizeLimit((int) (trainSet.size() * trainNeighbourhoodSizeLimitPercentage));
        }
    }

    private void setupTrainSetSize() {
        if (trainEstimateSetSizeLimitPercentage >= 0) {
            setTrainEstimateSetSizeLimit((int) (trainSet.size() * trainEstimateSetSizeLimitPercentage));
        }
    }

    private void setupTrainSetSampleStrategy() {
        switch (trainSetSampleStrategy) {
            case RANDOM:
                trainSetIterator = new RandomSampler(getTrainRandom().nextLong()); // todo make the remainder of these use seed instead of random
                break;
            case LINEAR:
                trainSetIterator = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                trainSetIterator = new RoundRobinRandomSampler(getTrainRandom());
                break;
            case DISTRIBUTED_RANDOM:
                trainSetIterator = new DistributedRandomSampler(getTrainRandom());
                break;
            default:
                throw new UnsupportedOperationException();
        }
        trainSetIterator.addAll(trainSet);
    }

    @Override
    public void copyFromSerObject(final Object obj) throws
                                                    Exception {
        super.copyFromSerObject(obj);
        Knn other = (Knn) obj;
        trainKNearestNeighbours.clear();
        for (KNearestNeighbours KNearestNeighbours : other.trainKNearestNeighbours) {
            trainKNearestNeighbours.add(new KNearestNeighbours(KNearestNeighbours));
        }
        setK(other.getK());
        setDistanceMeasure(DistanceMeasure.fromString(other.getDistanceMeasure().toString()));
        distanceMeasure.setOptions(other.getDistanceMeasure()
                                        .getOptions());
        setEarlyAbandon(other.getEarlyAbandon());
        setTrainNeighbourhoodSizeLimit(other.getTrainNeighbourhoodSizeLimit());
        setNeighbourSearchStrategy(other.getNeighbourSearchStrategy());
        setTrainEstimateSetSizeLimitPercentage(other.getTrainEstimateSetSizeLimitPercentage());
        setTrainEstimateSetSizeLimit(other.getTrainEstimateSetSizeLimit());
        setTrainNeighbourhoodSizeLimitPercentage(other.getTrainNeighbourhoodSizeLimitPercentage());
        setTrainSetSampleStrategy(other.getTrainSetSampleStrategy());
        trainEstimateSet.clear();
        trainEstimateSet.addAll(other.trainEstimateSet);
        trainNeighbourhood.clear();
        trainNeighbourhood.addAll(other.trainNeighbourhood);
        trainSetIterator = other.trainSetIterator.iterator();
        neighboursIterator = other.neighboursIterator.iterator();
        trainSet = other.trainSet;
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                         Exception {
        KNearestNeighbours testKNearestNeighbours = new KNearestNeighbours(testInstance);
        testKNearestNeighbours.addAll(trainSet); // todo limited test neighbourhood
        testKNearestNeighbours.trim();
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

    public enum TrainSetSampleStrategy {
        RANDOM,
        LINEAR,
        ROUND_ROBIN_RANDOM,
        DISTRIBUTED_RANDOM;

        public static TrainSetSampleStrategy fromString(String str) {
            for (TrainSetSampleStrategy s : values()) {
                if (s.name()
                        .equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    private static class Prediction {
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
