package timeseriesweka.classifiers.distance_based.ee;

import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.TrainAccuracyEstimator;
import timeseriesweka.classifiers.distance_based.ee.selection.BestPerTypeSelector;
import timeseriesweka.classifiers.distance_based.ee.selection.Selector;
import utilities.ArrayUtilities;
import utilities.iteration.AbstractIterator;
import utilities.iteration.linear.LinearIterator;
import utilities.iteration.linear.RoundRobinIterator;
import utilities.iteration.random.RandomIterator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class Ee extends AbstractClassifier {
    private List<Function<Instances, ParameterSpace>> parameterSpaceFunctions = new ArrayList<>();
    private List<Member> members = new ArrayList<>();
    private AbstractIterator<AbstractClassifier> candidateIterator;
    private AbstractIterator<AbstractClassifier> improvementIterator;
    private final Random trainRandom = new Random();
    private final Random testRandom = new Random();
    private Long trainSeed;
    private Long testSeed;
    private Selector<Benchmark> selector = new BestPerTypeSelector<>();
    private List<Benchmark> constituents;

    private static class Benchmark {
        private final AbstractClassifier classifier;
        private final ClassifierResults trainResults;

        private Benchmark(AbstractClassifier classifier, ClassifierResults trainResults) {
            this.classifier = classifier;
            this.trainResults = trainResults;
        }

        public ClassifierResults getTrainResults() {
            return trainResults;
        }

        public AbstractClassifier getClassifier() {
            return classifier;
        }
    }

    public List<Function<Instances, ParameterSpace>> getParameterSpaceFunctions() {
        return parameterSpaceFunctions;
    }

    public void setParameterSpaceFunctions(List<Function<Instances, ParameterSpace>> parameterSpaceFunctions) {
        this.parameterSpaceFunctions = parameterSpaceFunctions;
    }

    private static class Member {
        private final ParameterSpace parameterSpace;
        private final Iterator<Integer> indexIterator;

        private Member(ParameterSpace parameterSpace, Iterator<Integer> indexIterator) {
            this.parameterSpace = parameterSpace;
            this.indexIterator = indexIterator;
        }

        public Iterator<Integer> getIndexIterator() {
            return indexIterator;
        }

        public ParameterSpace getParameterSpace() {
            return parameterSpace;
        }
    }

    private void setup(Instances trainInstances) {
        // todo selector random
        candidateIterator = new LinearIterator<>();
        improvementIterator = new RoundRobinIterator<>();
        for(Function<Instances, ParameterSpace> function : parameterSpaceFunctions) {
            ParameterSpace parameterSpace = function.apply(trainInstances);
            parameterSpace.removeDuplicateParameterSets();
            if(parameterSpace.size() > 0) {
                AbstractIterator<Integer> iterator = new RandomIterator<>(); // todo set seed
                iterator.addAll(ArrayUtilities.sequence(parameterSpace.size()));
                members.add(new Member(parameterSpace, iterator));
            }
        }
    }

    private static class Abc {
        private final AbstractClassifier classifier;

    }

    private void nextCandidate(Instances trainInstances) throws Exception {
        AbstractClassifier classifier = candidateIterator.next();
        candidateIterator.remove();




        classifier.buildClassifier(trainInstances);
        ClassifierResults trainResults;
        if(classifier instanceof TrainAccuracyEstimator) {
            trainResults = ((TrainAccuracyEstimator) classifier).getTrainResults();
        } else {
            throw new UnsupportedOperationException(); // todo 10 fold cv
        }
        Benchmark benchmark = new Benchmark(classifier, trainResults);
        selector.add(benchmark);
//        improvementIterator.add(classifier);
    }

    private void nextImprovement(Instances trainInstances) {
        AbstractClassifier classifier = improvementIterator.next();
        improvementIterator.remove();
        // todo
    }

    private boolean hasRemainingCandidates() {
        return candidateIterator.hasNext();
    }

    private boolean hasRemainingImprovements() {
        return improvementIterator.hasNext();
    }

    @Override
    public void buildClassifier(Instances trainInstances) throws Exception {
        setup(trainInstances);
        boolean hasRemainingCandidates = hasRemainingCandidates();
        boolean hasRemainingImprovements = hasRemainingImprovements();
        while ((hasRemainingCandidates || hasRemainingImprovements)) {
            boolean choice = hasRemainingCandidates;
            if(hasRemainingCandidates && hasRemainingImprovements) {
                choice = trainRandom.nextBoolean();
            }
            if(choice) {
                nextCandidate(trainInstances);
            } else {
                nextImprovement(trainInstances);
            }
            hasRemainingCandidates = hasRemainingCandidates();
            hasRemainingImprovements = hasRemainingImprovements();
        }
        constituents = selector.getSelected();
    }

    @Override
    public double[] distributionForInstance(Instance testInstance) throws Exception {
        double[] distribution = new double[testInstance.numClasses()];
        for(Benchmark constituent : constituents) {
            double weight = constituent.getTrainResults().getAcc();
            double[] constituentDistribution = constituent.getClassifier().distributionForInstance(testInstance);
            ArrayUtilities.multiplyInPlace(constituentDistribution, weight);
            ArrayUtilities.addInPlace(distribution, constituentDistribution);
        }
        ArrayUtilities.normaliseInPlace(distribution);
        return distribution;
    }
}
