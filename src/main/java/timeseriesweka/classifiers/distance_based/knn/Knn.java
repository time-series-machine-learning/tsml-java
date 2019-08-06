//package timeseriesweka.classifiers.distance_based.knn;
//
//import evaluation.storage.ClassifierResults;
//import timeseriesweka.classifiers.Seedable;
//import timeseriesweka.classifiers.distance_based.distances.DistanceMeasure;
//import timeseriesweka.classifiers.distance_based.distances.dtw.Dtw;
//import timeseriesweka.config.AbstractConfig;
//import utilities.ArrayUtilities;
//import timeseriesweka.classifiers.distance_based.ee.selection.BestKSelector;
//import utilities.Copyable;
//import utilities.iteration.AbstractIterator;
//import weka.classifiers.AbstractClassifier;
//import weka.core.Capabilities;
//import weka.core.Instance;
//import weka.core.Instances;
//
//import java.util.*;
//
//public class Knn
//    extends AbstractClassifier
////            Seedable,
//
//    implements Copyable {
//
//    public static class Config
//        extends AbstractConfig {
//        // config options
//        private final static String K_KEY = "k";
//        private int k = 1;
//        private final static String DISTANCE_MEASURE_KEY = "dm";
//        private DistanceMeasure distanceMeasure = new Dtw();
//        private final static String EARLY_ABANDON_KEY = "ea";
//        private boolean earlyAbandon = false;
//
//        public Config() {
//
//        }
//
//        public Config(Config other) throws
//                                                                                        Exception {
//            super(other);
//        }
//
//        @Override
//        public void setOption(final String key, final String value) {
//            switch (key) {
//                case K_KEY:
//                    setK(Integer.parseInt(value));
//                    break;
//                case DISTANCE_MEASURE_KEY:
//                    setDistanceMeasure(DistanceMeasure.fromString(value));
//                    break;
//                case EARLY_ABANDON_KEY:
//                    setEarlyAbandon(Boolean.parseBoolean(value));
//                    break;
//            }
//        }
//
//        @Override
//        public void setOptions(final String[] options) throws
//                                                       Exception {
//            super.setOptions(options);
//            distanceMeasure.setOptions(options);
//        }
//
//        @Override
//        public String[] getOptions() {
//            return ArrayUtilities.concat(distanceMeasure.getOptions(), new String[] {
//                DISTANCE_MEASURE_KEY,
//                distanceMeasure.toString(),
//                K_KEY,
//                String.valueOf(k),
//                EARLY_ABANDON_KEY,
//                String.valueOf(earlyAbandon),
//                });
//        }
//
//        public int getK() {
//            return k;
//        }
//
//        public void setK(int k) {
//            this.k = k;
//        }
//
//        public DistanceMeasure getDistanceMeasure() {
//            return distanceMeasure;
//        }
//
//        public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
//            this.distanceMeasure = distanceMeasure;
//        }
//
//        public boolean isEarlyAbandon() {
//            return earlyAbandon;
//        }
//
//        public void setEarlyAbandon(final boolean earlyAbandon) {
//            this.earlyAbandon = earlyAbandon;
//        }
//
//        @Override
//        public Config copy() throws
//                                                                              Exception {
//            Config configuration = new Config();
//            configuration.copyFrom(this);
//            return configuration;
//        }
//
//        @Override
//        public void copyFrom(final Object object) throws
//                                                  Exception {
//            Config other = (Config) object;
//            setOptions(other.getOptions());
//        }
//
//    }
//
//    private Config config = new Config();
//    private AbstractIterator<Instance> trainSetIterator = null;
//    private AbstractIterator<Instance> trainEstimateSetIterator = null;
//    private Random trainRandom = new Random();
//
//    private List<NeighbourSearcher> trainEstimators = null;
//    private List<Instance> trainSet = null;
//    private List<Instance> trainNeighbourhood = null;
//
//    @Override
//    public void setOptions(String[] options) throws Exception {
//        getConfig().setOptions(options);
//    }
//
//    public Config getConfig() {
//        return config;
//    }
//
//    public Knn() {}
//
//    public Knn(Knn other) throws
//                          Exception {
//        copyFrom(other);
//    }
//
//    @Override
//    public String[] getOptions() {
//        return getConfig().getOptions();
//    }
//
//    @Override
//    public Capabilities getCapabilities() {
//        Capabilities result = super.getCapabilities();
//        result.disableAll();
//        // attributes must be numeric
//        // Here add in relational when ready
//        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
//        // class
//        result.enable(Capabilities.Capability.NOMINAL_CLASS);
//        // instances
//        result.setMinimumNumberInstances(0);
//        return result;
//    }
//
//    @Override
//    public String toString() {
//        return "KNN";
//    }
//
//    @Override
//    public void copyFrom(final Object object) throws
//                                              Exception {
//        Knn other = (Knn) object;
//        trainEstimators = new ArrayList<>();
//        for (NeighbourSearcher neighbourSearcher : other.trainEstimators) {
//            trainEstimators.add(new NeighbourSearcher(neighbourSearcher));
//        }
//        getConfig().copyFrom(other.getConfig());
//        trainNeighbourhood = new ArrayList<>(other.trainNeighbourhood);
//        trainSet = other.trainSet;
//    }
//
//    private boolean reset = true;
//
//    public void reset() {
//        reset = true;
//    }
//
//    private void setup(Instances trainSet) throws
//                                           Exception {
//        if(reset) {
//            reset = false;
//            this.trainSet = trainSet;
//            trainEstimators = new ArrayList<>();
//            trainNeighbourhood = new ArrayList<>();
//        }
//    }
//
//    private boolean hasRemainingTrainEstimations() {
//        return trainEstimateSetIterator.hasNext();
//    }
//
//    private boolean hasRemainingNeighbours() {
//        return trainSetIterator.hasNext();
//    }
//
//    private void nextTrainEstimator() {
//        Instance instance = trainEstimateSetIterator.next();
//        trainEstimateSetIterator.remove();
//        NeighbourSearcher neighbourSearcher = new NeighbourSearcher(instance);
//        trainEstimators.add(neighbourSearcher);
//        neighbourSearcher.addAll(trainNeighbourhood);
//    }
//
//    private static class Neighbour {
//        private final double distance;
//        private final Instance instance;
//
//        private Neighbour(final double distance, final Instance instance) {
//            this.distance = distance;
//            this.instance = instance;
//        }
//
//        public double getDistance() {
//            return distance;
//        }
//
//        public Instance getInstance() {
//            return instance;
//        }
//    }
//
//    private ClassifierResults trainResults;
//
//    private void buildTrainResults() throws Exception {
//        trainResults = new ClassifierResults();
//        for (NeighbourSearcher KNearestNeighbours : trainEstimators) {
//            KNearestNeighbours.trim();
//            Prediction prediction = KNearestNeighbours.predict();
//            double[] distribution = prediction.getDistribution();
//            trainResults.addPrediction(KNearestNeighbours.getTarget().classValue(),
//                    distribution,
//                    ArrayUtilities.max(distribution, getTrainRandom()),
//                    prediction.getPredictionTimeNanos(),
//                    null);
//        }
//        setClassifierResultsMetaInfo(trainResults);
//    }
//
//    @Override
//    public void buildClassifier(Instances trainSet) throws
//                                                           Exception {
//        setup(trainSet);
//        boolean remainingTrainEstimations = hasRemainingTrainEstimations();
//        boolean remainingNeighbours = hasRemainingNeighbours();
//        boolean choice = true;
//        while ((
//                    remainingTrainEstimations
//                    || remainingNeighbours
//                )
//                ) {
//            if(remainingTrainEstimations && remainingNeighbours) {
//                choice = !choice;
//            } else choice = !remainingNeighbours;
//            if(choice) {
//                nextTrainEstimator();
//            } else {
//                nextNeighbourSearch();
//            }
//            remainingTrainEstimations = hasRemainingTrainEstimations();
//            remainingNeighbours = hasRemainingNeighbours();
//        }
//        buildTrainResults();
//    }
//
//    public Knn copy() {
//        Knn knn = new Knn();
//        try {
//            knn.copyFrom(this);
//            return knn;
//        } catch (Exception e) {
//            throw new IllegalStateException(e);
//        }
//    }
//
//    private void nextNeighbourSearch() {
//        Instance trainNeighbour = trainSetIterator.next();
//        trainSetIterator.remove();
//        trainNeighbourhood.add(trainNeighbour);
//        for (BestKSelector<Instance, Double> trainEstimator : this.trainEstimators) {
//            trainEstimator.add(trainNeighbour);
//        }
//    }
//
//    @Override
//    public double[] distributionForInstance(final Instance testInstance) throws
//                                                                         Exception {
//        NeighbourSearcher testKNearestNeighbours = new NeighbourSearcher(testInstance);
//        testKNearestNeighbours.addAll(trainSet);
//        testKNearestNeighbours.trim();
//        return testKNearestNeighbours.predict().getDistribution();
//    }
//
//    private static class Prediction { // todo use time units, put into own class
//        private final double[] distribution;
//        private final long predictionTimeNanos;
//
//        private Prediction(double[] distribution, long predictionTimeNanos) {
//            this.distribution = distribution;
//            this.predictionTimeNanos = predictionTimeNanos;
//        }
//
//        public double[] getDistribution() {
//            return distribution;
//        }
//
//        public long getPredictionTimeNanos() {
//            return predictionTimeNanos;
//        }
//
//    }
//
//    private class NeighbourSearcher {
//
//        private final Instance target;
//        private final BestKSelector<Neighbour, Double> collector;
//        private long searchTimeNanos = 0;
//        private DistanceMeasure distanceMeasure;
//
//        private NeighbourSearcher(final Instance target) {
//            this.target = target;
//            collector = new BestKSelector<>();
//            collector.setComparator(Comparator.comparingDouble(aDouble -> aDouble));
//            collector.setLimit(config.getK());
//            collector.setExtractor(Neighbour::getDistance);
//            distanceMeasure = config.getDistanceMeasure();
//        }
//
//        public NeighbourSearcher(NeighbourSearcher other) throws
//                                                            Exception {
//            target = other.target;
//            collector = new BestKSelector<>(other.collector);
//            searchTimeNanos = other.searchTimeNanos;
//            distanceMeasure = other.distanceMeasure;
//        }
//
//        public void trim() {
//            long startTime = System.nanoTime();
//            int k = config.getK();
//            if(size <= k) {
//                trimmedKNeighbours = kNeighbours;
//            } else {
//                trimmedKNeighbours = new TreeMap<>();
//                List<Instance> furthestNeighbours = null;
//                for (Map.Entry<Double, Collection<Instance>> entry : kNeighbours.entrySet()) {
//                    furthestNeighbours = new ArrayList<>(entry.getValue());
//                    trimmedKNeighbours.put(entry.getKey(), furthestNeighbours);
//                }
//                if(furthestNeighbours != null) {
//                    int size = furthestNeighbours.size();
//                    while (size > k) {
//                        size--;
//                        int index = trainRandom.nextInt(furthestNeighbours.size());
//                        furthestNeighbours.remove(index);
//                    }
//                }
//            }
//            searchTimeNanos = System.nanoTime() - startTime;
//        }
//
//        public Instance getTarget() {
//            return target;
//        }
//
//        public Prediction predict() {
//            long startTime = System.nanoTime();
//            double[] distribution = new double[target.numClasses()];
//            TreeMap<Double, Collection<Instance>> neighbours = trimmedKNeighbours;
//            if (neighbours.size() == 0) {
//                distribution[getTestRandom().nextInt(distribution.length)]++;
//            } else {
//                for (Map.Entry<Double, Collection<Instance>> entry : neighbours.entrySet()) {
//                    for (Instance instance : entry.getValue()) {
//                        // todo weighted
//                        distribution[(int) instance.classValue()]++;
//                    }
//                }
//                ArrayUtilities.normaliseInplace(distribution);
//            }
//            long predictTimeNanos = System.nanoTime() - startTime;
//            return new Prediction(distribution, predictTimeNanos);
//        }
//
//        public void addAll(final List<Instance> instances) {
//            for (Instance instance : instances) {
//                add(instance);
//            }
//        }
//
//        public double add(Instance instance) {
//            long startTime = System.nanoTime();
//            if(!instance.equals(target)) {
//                double maxDistance = config.isEarlyAbandon() ?
//                                     collector.getLargestValue() :
//                                     Double.POSITIVE_INFINITY;
//                config.getDistanceMeasure().setLimit(maxDistance);
//                config.getDistanceMeasure().setCandidate(instance);
//                double distance = config.getDistanceMeasure().distance();
//                collector.add(new Neighbour(distance, instance));
//                searchTimeNanos += System.nanoTime() - startTime;
//                return distance;
//            } else {
//                return Double.POSITIVE_INFINITY;
//            }
//        }
//
//        public void add(Neighbour neighbour) {
//            long startTime = System.nanoTime();
//            if(!neighbour.getInstance().equals(target)) {
//                collector.add(neighbour);
//            }
//            searchTimeNanos += System.nanoTime() - startTime;
//        }
//    }
//
//    public ClassifierResults getTrainResults() {
//        throw new UnsupportedOperationException();
//    }
//}
