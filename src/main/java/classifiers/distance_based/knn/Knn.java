package classifiers.distance_based.knn;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.knn.sampling.*;
import classifiers.template_classifier.TemplateClassifier;
import classifiers.template_classifier.OptionSet.Option;
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

    // configuration options
    private final Option<Integer> kOption = getOptionSet().new Option<>(1, "k", Integer::parseInt);
    private final Option<DistanceMeasure> distanceMeasureOption = getOptionSet().new Option<>(new Dtw(0), "distanceMeasure", DistanceMeasure::fromString);
    private final Option<Boolean> earlyAbandonOption = getOptionSet().new Option<>(true, "ea", Boolean::parseBoolean);
    // restriction options
    private final Option<Integer> trainNeighbourhoodSizeLimitOption = getOptionSet().new Option<>(-1, "trnsl", Integer::parseInt);
    private final Option<Double> trainNeighbourhoodSizeLimitPercentageOption = getOptionSet().new Option<>(-1d, "trnslp", Double::parseDouble);
    private final Option<Integer> trainEstimateSetSizeLimitOption = getOptionSet().new Option<>(-1, "tressl", Integer::parseInt);
    private final Option<Double> trainEstimateSetSizeLimitPercentageOption = getOptionSet().new Option<>(-1d, "tresslp", Double::parseDouble);
    private final Option<Integer> testNeighbourhoodSizeLimitOption = getOptionSet().new Option<>(-1, "tensl", Integer::parseInt);
    private final Option<Double> testNeighbourhoodSizeLimitPercentageOption = getOptionSet().new Option<>(-1d, "tenslp", Double::parseDouble);
    // iteration options
    private final Option<NeighbourSearchStrategy> trainNeighbourSearchStrategyOption = getOptionSet().new Option<>(NeighbourSearchStrategy.RANDOM, "trnss", NeighbourSearchStrategy::fromString);
    private final Option<TrainEstimationSource> trainEstimationSourceOption = getOptionSet().new Option<>(TrainEstimationSource.FROM_TRAIN_SET, "trsss", TrainEstimationSource::fromString);
    private final Option<TrainEstimationStrategy> trainEstimationStrategyOption = getOptionSet().new Option<>(TrainEstimationStrategy.RANDOM, "tres", TrainEstimationStrategy::fromString);
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
    public void setOption(String key, String value) {
        try {
            super.setOption(key, value);
        } catch (IllegalArgumentException e) {
            distanceMeasureOption.set(value);
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

    public double getTestNeighbourhoodSizeLimitPercentageOption() {
        return testNeighbourhoodSizeLimitPercentageOption.get();
    }
    public void setTestNeighbourhoodSizeLimitPercentageOption(double testNeighbourhoodSizeLimitPercentageOption) {
        this.testNeighbourhoodSizeLimitPercentageOption.set(testNeighbourhoodSizeLimitPercentageOption);
    }

    public Option<TrainEstimationStrategy> getTrainEstimationStrategyOption() {
        return trainEstimationStrategyOption;
    }

    public int getTestNeighbourhoodSizeLimitOption() {
        return testNeighbourhoodSizeLimitOption.get();
    }
    public void setTestNeighbourhoodSizeLimitOption(int testNeighbourhoodSizeLimitOption) {
        this.testNeighbourhoodSizeLimitOption.set(testNeighbourhoodSizeLimitOption);
    }

    public Option<Integer> getkOption() {
        return kOption;
    }

    public Option<DistanceMeasure> getDistanceMeasureOption() {
        return distanceMeasureOption;
    }

    public Option<Boolean> getEarlyAbandonOption() {
        return earlyAbandonOption;
    }

    public Option<Integer> getTrainNeighbourhoodSizeLimitOption() {
        return trainNeighbourhoodSizeLimitOption;
    }

    public Option<Double> getTrainNeighbourhoodSizeLimitPercentageOption() {
        return trainNeighbourhoodSizeLimitPercentageOption;
    }

    public Option<Integer> getTrainEstimateSetSizeLimitOption() {
        return trainEstimateSetSizeLimitOption;
    }

    public Option<Double> getTrainEstimateSetSizeLimitPercentageOption() {
        return trainEstimateSetSizeLimitPercentageOption;
    }

    public Option<NeighbourSearchStrategy> getTrainNeighbourSearchStrategyOption() {
        return trainNeighbourSearchStrategyOption;
    }

    public Option<TrainEstimationSource> getTrainEstimationSourceOption() {
        return trainEstimationSourceOption;
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
        switch (trainNeighbourSearchStrategyOption.get()) {
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
        if (trainNeighbourhoodSizeLimitPercentageOption.get() >= 0) {
            trainNeighbourhoodSizeLimitOption.set((int) (trainSet.size() * trainNeighbourhoodSizeLimitPercentageOption.get()));
        }
    }

    private void setupTrainEstimateSetSize() {
        if (trainEstimateSetSizeLimitPercentageOption.get() >= 0) {
            trainEstimateSetSizeLimitOption.set((int) (trainSet.size() * trainEstimateSetSizeLimitPercentageOption.get()));
        }
    }

    private void setupTrainEstimationStrategy() {
        switch (trainEstimationStrategyOption.get()) {
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
            switch (trainEstimationSourceOption.get()) {
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
                    trainEstimate.size() < trainEstimateSetSizeLimitOption.get() // if train estimate set under limit
                    || trainEstimateSetSizeLimitOption.get() < 0 // if train estimate limit set
               )
               && trainEstimatorIterator.hasNext(); // if remaining train estimators
    }

    private boolean hasRemainingNeighbours() {
        return (
                    trainNeighbourhood.size() < trainNeighbourhoodSizeLimitOption.get() // if within train neighbourhood size limit
                    || trainNeighbourhoodSizeLimitOption.get() < 0 // if active train neighbourhood size limit
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
        if(trainEstimationSourceOption.get() == TrainEstimationSource.FROM_TRAIN_NEIGHBOURHOOD) {
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
            } else if(remainingNeighbours) {
                choice = false;
            } else {
                choice = true;
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
        getkOption().set(other.getkOption().get());
        getDistanceMeasureOption().set(DistanceMeasure.fromString(other.getDistanceMeasureOption().toString()));
//        distanceMeasureOption.setOptions(other.getDistanceMeasureOption()
//                                        .getOptions()); todo!
//        setEarlyAbandon(other.getEarlyAbandonOption());
//        setTrainNeighbourhoodSizeLimit(other.getTrainNeighbourhoodSizeLimitOption());
//        setTrainNeighbourSearchStrategy(other.getTrainNeighbourSearchStrategyOption());
//        setTrainEstimateSetSizeLimitPercentage(other.getTrainEstimateSetSizeLimitPercentageOption());
//        setTrainEstimateSetSizeLimit(other.getTrainEstimateSetSizeLimitOption());
//        setTrainNeighbourhoodSizeLimitPercentage(other.getTrainNeighbourhoodSizeLimitPercentageOption());
//        setTrainSetSampleStrategy(other.getTrainEstimationSourceOption());
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
            if(size <= kOption.get()) {
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
                    while (size > kOption.get()) {
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
            double maxDistance = earlyAbandonOption.get() ? furthestDistance : Double.POSITIVE_INFINITY;
            double distance = distanceMeasureOption.get().distance(target, instance, maxDistance);
            searchTimeNanos += System.nanoTime() - startTime;
            add(instance, distance);
            return distance;
        }

        public void add(Instance instance, double distance) {
            long startTime = System.nanoTime();
            if(!instance.equals(target)) {
                if((distance <= furthestDistance || size < kOption.get()) && kOption.get() > 0) {
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
                    if(distance < furthestDistance && size > kOption.get()) { // if we've got too many neighbours AND just added a neighbour closer than the furthest then try and knock off the furthest lot
                        int numFurthestNeighbours = furthestNeighbours.size();
                        if (size - kOption.get() >= numFurthestNeighbours) {
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
