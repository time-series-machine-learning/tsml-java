package classifiers.distance_based.knn;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import classifiers.template.classifier.TemplateClassifier;
import classifiers.template.config.ConfigState;
import classifiers.template.config.reduced.ReductionConfig;
import evaluation.storage.ClassifierResults;
import utilities.ArrayUtilities;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class Knn
    extends TemplateClassifier {

//    private final ConfigState<KnnConfig> knnConfigState = new ConfigState<>(KnnConfig::new,
//                                                                            KnnConfig.TRAIN_CONFIG_COMPARATOR,
//                                                                            KnnConfig.TEST_CONFIG_COMPARATOR);
//    private final ConfigState<ReductionConfig> reducedTrainSetConfigState =
//        new ConfigState<>(ReductionConfig::new, ReductionConfig.TRAIN_CONFIG_COMPARATOR,
//                          ReductionConfig.TEST_CONFIG_COMPARATOR);
    private KnnConfig knnConfig = new KnnConfig();
    private AbstractIterator<Instance> trainSetIterator = null;
    private AbstractIterator<Instance> trainEstimateSetIterator = null;

    // sets
    private List<KNearestNeighbours> trainEstimate = null;
    private List<Instance> trainSet = null;
    private List<Instance> trainNeighbourhood = null;

    @Override
    public void setOptions(String[] options) throws
                                                                Exception {
        getKnnConfig().setOptions(options);
    }


    @Override
    public void setTrainSeed(final Long seed) {
        super.setTrainSeed(seed);
        getKnnConfig().getReductionConfig().setSamplingSeed(seed);
    }

    public KnnConfig getKnnConfig() {
        return knnConfig;//State.getNext();
    }

    public Knn() {}

    public Knn(Knn other) throws
                          Exception {
        super(other);
    }


    @Override
    public String[] getOptions() {
        return ArrayUtilities.concat(getKnnConfig().getOptions(), super.getOptions());
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("haven't done this yet!");
    }

    @Override
    public String toString() {
        return "KNN";
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
        getKnnConfig().copyFrom(other.getKnnConfig());
        trainNeighbourhood = new ArrayList<>(other.trainNeighbourhood);
        trainSet = other.trainSet;
    }

    private void setup(Instances trainSet) throws
                                           Exception {
//        for (int i = 0; i < trainSet.size(); i++) {
//            trainSet.get(i)
//                    .setWeight(i);
//        }
//        knnConfigState.shift();
//        reducedTrainSetConfigState.shift();
        if (trainSetChanged(trainSet)
//                || knnConfigState.mustResetTrain() || reducedTrainSetConfigState.mustResetTrain()
        ) {
            getTrainStopWatch().reset();
            resetTrainSeed();
//            knnConfig = knnConfigState.getCurrent();
            this.trainSet = trainSet;
            trainEstimate = new ArrayList<>();
            trainNeighbourhood = new ArrayList<>();
            ReductionConfig reductionConfig = knnConfig.getReductionConfig();
//            ReductionConfig reductionConfig = reducedTrainSetConfigState.getCurrent();
            reductionConfig.buildTrainIterators(trainSet);
            trainSetIterator = reductionConfig.getTrainSetIterator();
            trainEstimateSetIterator = reductionConfig.getTrainEstimateSetIterator();
            getTrainStopWatch().lap();
        }
    }

    private boolean hasRemainingTrainEstimations() {
        return trainEstimateSetIterator.hasNext(); // if remaining train estimators
    }

    private boolean hasRemainingNeighbours() {
        return trainSetIterator.hasNext(); // if there are remaining neighbours
    }

    private void nextTrainEstimator() {
        Instance instance = trainEstimateSetIterator.next();
//        System.out.println(instance.weight());
        trainEstimateSetIterator.remove();
        KNearestNeighbours kNearestNeighbours = new KNearestNeighbours(instance);
        trainEstimate.add(kNearestNeighbours);
        kNearestNeighbours.addAll(trainNeighbourhood);
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

    private void nextNeighbourSearch() {
        Instance trainNeighbour = trainSetIterator.next();
        trainSetIterator.remove();
        trainNeighbourhood.add(trainNeighbour);
        for (KNearestNeighbours trainEstimator : this.trainEstimate) {
            trainEstimator.add(trainNeighbour);
        }
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                         Exception {
        KNearestNeighbours testKNearestNeighbours = new KNearestNeighbours(testInstance);
        testKNearestNeighbours.addAll(trainSet);
        testKNearestNeighbours.trim();
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
            int k = knnConfig.getK();
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
            double maxDistance = knnConfig.isEarlyAbandon() ?
                                 furthestDistance :
                                 Double.POSITIVE_INFINITY;
            double distance = knnConfig.getDistanceMeasure()
                                       .distance(target, instance, maxDistance);
            searchTimeNanos += System.nanoTime() - startTime;
            add(instance, distance);
            return distance;
        }

        public void add(Instance instance, double distance) {
            long startTime = System.nanoTime();
            if(!instance.equals(target)) {
                int k = knnConfig.getK();
                if ((distance <= furthestDistance || (size < k || k < 0))) {
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
                    if (distance < furthestDistance && size > k && k > 0) { // if we've got too many neighbours AND just added a neighbour closer than the furthest then try and knock off the furthest lot
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
