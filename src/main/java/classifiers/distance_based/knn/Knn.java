package classifiers.distance_based.knn;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.knn.sampling.*;
import classifiers.template.TemplateClassifier;
import classifiers.template.configuration.ConfigState;
import evaluation.storage.ClassifierResults;
import utilities.ArrayUtilities;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class Knn
    extends TemplateClassifier<Knn> {

    public KnnConfig getConfig() {
        return configState.getNextConfig();
    }

    private final ConfigState<KnnConfig> configState = new ConfigState<>(KnnConfig::new);
    private KnnConfig config = null;
    // sets
    private List<KNearestNeighbours> trainEstimate = null;
    private List<Instance> trainSet = null;
    private List<Instance> trainNeighbourhood = null;
    // iterators for executing strategies
    private DynamicIterator<Instance, ?> trainNeighbourIterator = null;
    private DynamicIterator<Instance, ?> trainEstimatorIterator = null;

    // todo if k <= 0 then use all neighbours

    public Knn() {}

    public Knn(Knn other) throws
                          Exception {
        super(other);
    }

    @Override
    public void setOption(final String key, final String value) throws
                                                                Exception {
        config.setOption(key, value);
    }

    @Override
    public String[] getOptions() {
        return config.getOptions();
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("haven't done this yet!");
    }

    @Override
    public String toString() {
        return "KNN";
    }

    private void setup(Instances trainSet) throws
                                           Exception {
        configState.shift();
        if (trainSetChanged(trainSet) || configState.mustResetTrain()) {
            getTrainStopWatch().reset();
            config = configState.getCurrentConfig();
            this.trainSet = trainSet;
            trainEstimate = new ArrayList<>();
            trainNeighbourhood = new ArrayList<>();
            config.setupNeighbourhoodSize(trainSet);
            config.setupTrainEstimateSetSize(trainSet);
            config.setupTrainNeighbourhoodSizeThreshold(trainSet);
            trainNeighbourIterator = buildNeighbourSearchStrategy(trainSet, getTrainRandom());
            trainEstimatorIterator = buildTrainEstimationStrategy(trainSet, getTestRandom());
        }
    }


    public DynamicIterator<Instance, ?> buildNeighbourSearchStrategy(Collection<Instance> trainSet, Random random) {
        DynamicIterator<Instance, ?> iterator;
        switch (config.getTrainNeighbourSearchStrategy()) {
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
        if(config.getPredefinedTrainNeighbourhood() != null) {
            iterator.addAll(config.getPredefinedTrainNeighbourhood());
        } else {
            iterator.addAll(trainSet);
        }
        return iterator;
    }

    public DynamicIterator<Instance, ?> buildTrainEstimationStrategy(Collection<Instance> trainSet, Random random) {
        DynamicIterator<Instance, ?> iterator;
        switch (config.getTrainEstimationStrategy()) {
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
        if(config.getPredefinedTrainEstimateSet() != null) {
            iterator.addAll(config.getPredefinedTrainEstimateSet());
        } else {
            switch (config.getTrainEstimationSource()) {
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

    private boolean hasRemainingTrainEstimations() {
        return (
                   trainEstimate.size() < config.getTrainEstimateSetSizeLimit() // if train estimate set under limit
                   && config.hasTrainEstimateSetSizeLimit() // if train estimate limit set
               )
               && trainEstimatorIterator.hasNext(); // if remaining train estimators
    }

    private boolean hasRemainingNeighbours() {
        return (
                   trainNeighbourhood.size() < config.getTrainNeighbourhoodSizeLimit() // if within train neighbourhood size limit
                   && config.hasTrainNeighbourhoodSizeLimit() // if active train neighbourhood size limit
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
        if(config.getTrainEstimationSource() == TrainEstimationSource.FROM_TRAIN_NEIGHBOURHOOD) {
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
        setup(trainSet);
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
    public void copyFrom(final Object object) throws
                                                    Exception {
        super.copyFrom(object);
        Knn other = (Knn) object;
        trainEstimate = new ArrayList<>();
        for (KNearestNeighbours KNearestNeighbours : other.trainEstimate) {
            trainEstimate.add(new KNearestNeighbours(KNearestNeighbours));
        }
        config.copyFrom(other.config);
        trainNeighbourhood = new ArrayList<>(other.trainNeighbourhood);
        trainEstimatorIterator = other.trainEstimatorIterator.iterator();
        trainNeighbourIterator = other.trainNeighbourIterator.iterator();
        trainSet = other.trainSet;
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                         Exception {
        KNearestNeighbours testKNearestNeighbours = new KNearestNeighbours(testInstance);
        testKNearestNeighbours.addAll(trainSet); // todo limited test neighbourhood
        testKNearestNeighbours.trim(); // todo empty train set should be rand predict
        return testKNearestNeighbours.predict().getDistribution();
    }

    private static class Prediction { // todo use time units, put into own class
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
            int k = config.getK();
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
            double maxDistance = config.isEarlyAbandon() ? furthestDistance : Double.POSITIVE_INFINITY;
            double distance = config.getDistanceMeasure().distance(target, instance, maxDistance);
            searchTimeNanos += System.nanoTime() - startTime;
            add(instance, distance);
            return distance;
        }

        public void add(Instance instance, double distance) {
            long startTime = System.nanoTime();
            if(!instance.equals(target)) {
                int k = config.getK();
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
