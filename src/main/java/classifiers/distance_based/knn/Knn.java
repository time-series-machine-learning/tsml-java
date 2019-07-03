package classifiers.distance_based.knn;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.knn.sampling.*;
import classifiers.template_classifier.TemplateClassifier;
import classifiers.template_classifier.OptionSet.Option;
import distances.DistanceMeasure;
import distances.time_domain.dtw.Dtw;
import evaluation.storage.ClassifierResults;
import utilities.ArrayUtilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class Knn
    extends TemplateClassifier {

    // configuration options
    private final Option<Integer> k = getOptionSet().new Option<>(1, "k");
    private final Option<DistanceMeasure> distanceMeasure = getOptionSet().new Option<>(new Dtw(0), "dm");
    private final Option<Boolean> earlyAbandon = getOptionSet().new Option<>(true, "ea");
    // restriction options
    private final Option<Integer> trainNeighbourhoodSizeLimit = getOptionSet().new Option<>(-1, "trnsl");
    private final Option<Double> trainNeighbourhoodSizeLimitPercentage = getOptionSet().new Option<>(-1d, "trnslp");
    private final Option<Integer> trainEstimateSetSizeLimit = getOptionSet().new Option<>(-1, "tressl");
    private final Option<Double> trainEstimateSetSizeLimitPercentage = getOptionSet().new Option<>(-1d, "tresslp");
    private final Option<Integer> testNeighbourhoodSizeLimit = getOptionSet().new Option<>(-1, "tensl");
    private final Option<Double> testNeighbourhoodSizeLimitPercentage = getOptionSet().new Option<>(-1d, "tenslp");
    // iteration options
    private final Option<NeighbourSearchStrategy> trainNeighbourSearchStrategy = getOptionSet().new Option<>(NeighbourSearchStrategy.RANDOM, "trnss");
    private final Option<TrainSetSampleStrategy> trainSetSampleStrategy = getOptionSet().new Option<>(TrainSetSampleStrategy.RANDOM, "trsss");

    public TrainEstimationStrategy getTrainEstimationStrategy() { // todo should it just pass through option?
        return trainEstimationStrategy.get();
    }

    public void setTrainEstimationStrategy(TrainEstimationStrategy trainEstimationStrategy) {
        this.trainEstimationStrategy.set(trainEstimationStrategy);
    }

    private final Option<TrainEstimationStrategy> trainEstimationStrategy = getOptionSet().new Option<>(TrainEstimationStrategy.FROM_TRAIN_SET, "tres");
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

    public Knn(Knn other) {// todo
        throw new UnsupportedOperationException();
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
        return testNeighbourhoodSizeLimitPercentage.get();
    }
    public void setTestNeighbourhoodSizeLimitPercentage(double testNeighbourhoodSizeLimitPercentage) {
        this.testNeighbourhoodSizeLimitPercentage.set(testNeighbourhoodSizeLimitPercentage);
    }

    public int getTestNeighbourhoodSizeLimit() {
        return testNeighbourhoodSizeLimit.get();
    }
    public void setTestNeighbourhoodSizeLimit(int testNeighbourhoodSizeLimit) {
        this.testNeighbourhoodSizeLimit.set(testNeighbourhoodSizeLimit);
    }

    public int getK() {
        return k.get();
    }
    public void setK(final int k) {
        this.k.set(k);
    }

    public TrainSetSampleStrategy getTrainSetSampleStrategy() {
        return trainSetSampleStrategy.get();
    }
    public void setTrainSetSampleStrategy(TrainSetSampleStrategy trainSetSampleStrategy) {
        this.trainSetSampleStrategy.set(trainSetSampleStrategy);
    }

    public String[] getOptions() { // todo
        throw new UnsupportedOperationException();
//        return ArrayUtilities.concat(super.getOptions(), new String[] {
//            K_KEY,
//            String.valueOf(k),
//                TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY,
//            String.valueOf(trainNeighbourhoodSizeLimit),
//            NEIGHBOUR_SEARCH_STRATEGY_KEY,
//            trainNeighbourSearchStrategy.name(),
//                TRAIN_ESTIMATE_SET_SIZE_KEY,
//            String.valueOf(trainEstimateSetSizeLimit),
//            EARLY_ABANDON_KEY,
//            String.valueOf(earlyAbandon),
//                TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY,
//            String.valueOf(trainNeighbourhoodSizeLimitPercentage),
//                TRAIN_ESTIMATE_SET_SIZE_PERCENTAGE_KEY,
//            String.valueOf(trainEstimateSetSizeLimitPercentage),
//            DISTANCE_MEASURE_KEY,
//            distanceMeasure.toString(),
//            StringUtilities.join(",", distanceMeasure.getOptions()),
//            });
    } // todo setOption and getOption in template classifier
    public void setOptions(final String[] options) throws
                                                   Exception { // todo
        throw new UnsupportedOperationException();
//        super.setOptions(options);
//        for (int i = 0; i < options.length - 1; i += 2) {
//            String key = options[i];
//            String value = options[i + 1];
//            if (key.equals(K_KEY)) {
//                setK(Integer.parseInt(value));
//            } else if (key.equals(TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_KEY)) {
//                setK(Integer.parseInt(value));
//            } else if (key.equals(DISTANCE_MEASURE_KEY)) {
//                setDistanceMeasure(DistanceMeasure.fromString(value));
//            } else if (key.equals(NEIGHBOUR_SEARCH_STRATEGY_KEY)) {
//                setTrainNeighbourSearchStrategy(NeighbourSearchStrategy.fromString(value));
//            } else if (key.equals(EARLY_ABANDON_KEY)) {
//                setEarlyAbandon(Boolean.parseBoolean(value));
//            } else if (key.equals(TRAIN_ESTIMATE_SET_SIZE_KEY)) {
//                setTrainEstimateSetSizeLimit(Integer.parseInt(value));
//            } else if (key.equals(TRAIN_ESTIMATE_SET_SIZE_PERCENTAGE_KEY)) {
//                setTrainEstimateSetSizeLimitPercentage(Double.parseDouble(value));
//            } else if (key.equals(TRAIN_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY)) {
//                setTrainNeighbourhoodSizeLimitPercentage(Double.valueOf(value));
//            } else if(key.equals(TEST_NEIGHBOURHOOD_SIZE_LIMIT_KEY)) {
//                setTestNeighbourhoodSizeLimit(Integer.parseInt(value));
//            } else if(key.equals(TEST_NEIGHBOURHOOD_SIZE_LIMIT_PERCENTAGE_KEY)) {
//                setTestNeighbourhoodSizeLimitPercentage(Double.parseDouble(value));
//            }
//        }
//        distanceMeasure.setOptions(options);
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure.get();
    }
    public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
        this.distanceMeasure.set(distanceMeasure);
    }

    public boolean getEarlyAbandon() {
        return earlyAbandon.get();
    }
    public void setEarlyAbandon(final boolean earlyAbandon) {
        this.earlyAbandon.set(earlyAbandon);
    }

    public int getTrainNeighbourhoodSizeLimit() {
        return trainNeighbourhoodSizeLimit.get();
    }
    public void setTrainNeighbourhoodSizeLimit(final int trainNeighbourhoodSizeLimit) {
        this.trainNeighbourhoodSizeLimit.set(trainNeighbourhoodSizeLimit);
    }

    public NeighbourSearchStrategy getTrainNeighbourSearchStrategy() {
        return trainNeighbourSearchStrategy.get();
    }
    public void setTrainNeighbourSearchStrategy(final NeighbourSearchStrategy trainNeighbourSearchStrategy) {
        this.trainNeighbourSearchStrategy.set(trainNeighbourSearchStrategy);
    }

    public double getTrainEstimateSetSizeLimitPercentage() {
        return trainEstimateSetSizeLimitPercentage.get();
    }
    public void setTrainEstimateSetSizeLimitPercentage(final double trainEstimateSetSizeLimitPercentage) {
        this.trainEstimateSetSizeLimitPercentage.set(trainEstimateSetSizeLimitPercentage);
    }

    public int getTrainEstimateSetSizeLimit() {
        return trainEstimateSetSizeLimit.get();
    }
    public void setTrainEstimateSetSizeLimit(final int trainEstimateSetSizeLimit) {
        this.trainEstimateSetSizeLimit.set(trainEstimateSetSizeLimit);
    }

    public double getTrainNeighbourhoodSizeLimitPercentage() {
        return trainNeighbourhoodSizeLimitPercentage.get();
    }
    public void setTrainNeighbourhoodSizeLimitPercentage(double trainNeighbourhoodSizeLimitPercentage) {
        this.trainNeighbourhoodSizeLimitPercentage.set(trainNeighbourhoodSizeLimitPercentage);
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
            setupNeighbourhoodSize();
            setupTrainEstimateSetSize();
            setupTrainEstimationStrategy();
            setupNeighbourSearchStrategy();
        }
    }

    private void setupNeighbourSearchStrategy() {
        switch (trainNeighbourSearchStrategy.get()) {
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
            trainNeighbourIterator.addAll(trainNeighbourhood);
        } else {
            trainNeighbourIterator.addAll(trainSet);
        }
    }

    private void setupNeighbourhoodSize() {
        if (trainNeighbourhoodSizeLimitPercentage.get() >= 0) {
            setTrainNeighbourhoodSizeLimit((int) (trainSet.size() * trainNeighbourhoodSizeLimitPercentage.get()));
        }
    }

    private void setupTrainEstimateSetSize() {
        if (trainEstimateSetSizeLimitPercentage.get() >= 0) {
            setTrainEstimateSetSizeLimit((int) (trainSet.size() * trainEstimateSetSizeLimitPercentage.get()));
        }
    }

    private void setupTrainEstimationStrategy() {
        switch (trainSetSampleStrategy.get()) {
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
            switch (trainEstimationStrategy.get()) {
                case FROM_TRAIN_SET:
                    trainEstimatorIterator.addAll(trainSet);
                    break;
                case FROM_TRAIN_NEIGHBOURHOOD:
                    throw new UnsupportedOperationException();
            }
        }
    }


    private boolean hasRemainingTrainEstimations() {
        return (
                    trainEstimate.size() < trainEstimateSetSizeLimit.get() // if train estimate set under limit
                    || trainEstimateSetSizeLimit.get() < 0 // if train estimate limit set
               )
               && trainEstimatorIterator.hasNext(); // if remaining train estimators
    }

    private boolean hasRemainingNeighbours() {
        return (
                    trainNeighbourhood.size() < trainNeighbourhoodSizeLimit.get() // if within train neighbourhood size limit
                    || trainNeighbourhoodSizeLimit.get() < 0 // if active train neighbourhood size limit
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
            } else if(remainingNeighbours) {
                choice = false;
            }
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

    public Knn copy() {
        Knn knn = new Knn();
        try {
            knn.copyFrom(this);
            return knn;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void copyFromSerObject(final Object obj) throws
                                                    Exception {
        super.copyFromSerObject(obj);
        Knn other = (Knn) obj;
        trainEstimate.clear();
        for (KNearestNeighbours KNearestNeighbours : other.trainEstimate) {
            trainEstimate.add(new KNearestNeighbours(KNearestNeighbours));
        }
        setK(other.getK());
        setDistanceMeasure(DistanceMeasure.fromString(other.getDistanceMeasure().toString()));
//        distanceMeasure.setOptions(other.getDistanceMeasure()
//                                        .getOptions()); todo!
        setEarlyAbandon(other.getEarlyAbandon());
        setTrainNeighbourhoodSizeLimit(other.getTrainNeighbourhoodSizeLimit());
        setTrainNeighbourSearchStrategy(other.getTrainNeighbourSearchStrategy());
        setTrainEstimateSetSizeLimitPercentage(other.getTrainEstimateSetSizeLimitPercentage());
        setTrainEstimateSetSizeLimit(other.getTrainEstimateSetSizeLimit());
        setTrainNeighbourhoodSizeLimitPercentage(other.getTrainNeighbourhoodSizeLimitPercentage());
        setTrainSetSampleStrategy(other.getTrainSetSampleStrategy());
        trainNeighbourhood.clear();
        trainNeighbourhood.addAll(other.trainNeighbourhood);
        trainEstimatorIterator = other.trainEstimatorIterator.iterator();
        trainNeighbourIterator = other.trainNeighbourIterator.iterator();
        trainSet = other.trainSet;
        throw new UnsupportedOperationException();
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

    public enum TrainEstimationStrategy {
        FROM_TRAIN_NEIGHBOURHOOD,
        FROM_TRAIN_SET,
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
            if(size <= k.get()) {
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
                    while (size > k.get()) {
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
            double maxDistance = earlyAbandon.get() ? furthestDistance : Double.POSITIVE_INFINITY;
            double distance = distanceMeasure.get().distance(target, instance, maxDistance);
            searchTimeNanos += System.nanoTime() - startTime;
            add(instance, distance);
            return distance;
        }

        public void add(Instance instance, double distance) {
            long startTime = System.nanoTime();
            if(!instance.equals(target)) {
                if((distance <= furthestDistance || size < k.get()) && k.get() > 0) {
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
                    if(distance < furthestDistance && size > k.get()) { // if we've got too many neighbours AND just added a neighbour closer than the furthest then try and knock off the furthest lot
                        int numFurthestNeighbours = furthestNeighbours.size();
                        if (size - k.get() >= numFurthestNeighbours) {
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
