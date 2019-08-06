package timeseriesweka.classifiers.distance_based.ee;

import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.distance_based.distances.DistanceMeasure;
import timeseriesweka.classifiers.distance_based.ee.selection.BestPerTypeSelector;
import timeseriesweka.classifiers.distance_based.ee.selection.Selector;
import utilities.ArrayUtilities;
import utilities.iteration.ParameterSetIterator;
import utilities.iteration.AbstractIterator;
import utilities.iteration.random.RandomIterator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

public class Ee extends AbstractClassifier {

    private List<Function<Instances, ParameterSpace>> parameterSpaceFunctions = new ArrayList<>();
//    private Selector<Candidate> selector = new BestPerTypeSelector<Candidate, ParameterSpace>(Candidate::getParameterSpace, Comparator.comparingDouble(candidate -> candidate.getTrainResults().getAcc()));

    private List<Candidate> constituents;
    private int numClasses;
    private AbstractIterator<AbstractIterator<ParameterSet>> parameterSetIteratorIterator;
    private Long seed;
    private Selector<Candidate> selector = new BestPerTypeSelector<>(bestPerDistanceMeasureTraditional(),
                                                                     Comparator.comparingDouble(candidate -> candidate.getTrainResults().getAcc()));

    private static Function<Candidate, String> bestPerDistanceMeasure() {
        return candidate -> candidate.getParameterSet()
                                     .getParameterValue(DistanceMeasure.DISTANCE_MEASURE_KEY);
    }

    private static Function<Candidate, String> bestPerDistanceMeasureTraditional() {
        return new Function<Candidate, String>() {
            @Override
            public String apply(final Candidate candidate) {
//                Dtw.NAME;
                return null;
            }
        };
    }

    private void setup(Instances trainingSet) {
        if(seed == null) {
            throw new IllegalStateException("seed not set");
        }
        numClasses = trainingSet.numClasses();
        buildParameterSetIterators(trainingSet);
    }

    private void buildParameterSpaceIterator() {
        parameterSetIteratorIterator = new RandomIterator<>(seed);
    }

    private void buildParameterSetIterators(Instances trainingSet) {
        for(Function<Instances, ParameterSpace> function : parameterSpaceFunctions) {
            ParameterSpace parameterSpace = function.apply(trainingSet);
            if(parameterSpace.isEmpty()) {
                continue;
            }
            parameterSpace.removeDuplicateParameterSets();
            RandomIterator<Integer> randomIterator = new RandomIterator<>(seed);
            randomIterator.addAll(ArrayUtilities.sequence(parameterSpace.size()));
            ParameterSetIterator parameterSetIterator = new ParameterSetIterator(parameterSpace, randomIterator);
            parameterSetIteratorIterator.add(parameterSetIterator);
        }
    }

    @Override
    public void buildClassifier(final Instances trainingSet) throws
                                                      Exception {
        setup(trainingSet);
//        while (parameterSetIteratorIterator.hasNext()) {
//            AbstractIterator<ParameterSet> parameterSetIterator = parameterSetIteratorIterator.next();
//            ParameterSet parameterSet = parameterSetIterator.next();
//            parameterSetIterator.remove();
//            if(!parameterSetIterator.hasNext()) {
//                parameterSetIteratorIterator.remove();
//            }
//            Knn knn = new Knn();
//            knn.setOptions(parameterSet.getOptions());
//            ClassifierResults trainResults = knn.getTrainResults();
//            Candidate candidate = new Candidate(knn, parameterSet, trainResults);
//            selector.add(candidate);
//        }
//        constituents = selector.getSelected();
    }

    @Override
    public double[] distributionForInstance(final Instance testCase) throws
                                                                     Exception {
        double[] overallDistribution = new double[numClasses];
        for(Candidate constituent : constituents) {
            double[] distribution = constituent.getClassifier()
                                          .distributionForInstance(testCase);
            ArrayUtilities.addInPlace(overallDistribution, distribution);
        }
        ArrayUtilities.normaliseInPlace(overallDistribution);
        return overallDistribution;
    }

    private static class Candidate {
        private final AbstractClassifier classifier;
        private final ParameterSet parameterSet;
        private final ClassifierResults trainResults;

        private Candidate(final AbstractClassifier classifier,
                          final ParameterSet parameterSet, final ClassifierResults trainResults) {
            this.classifier = classifier;
            this.parameterSet = parameterSet;
            this.trainResults = trainResults;
        }

        public AbstractClassifier getClassifier() {
            return classifier;
        }


        public ClassifierResults getTrainResults() {
            return trainResults;
        }

        public ParameterSet getParameterSet() {
            return parameterSet;
        }
    }
}
