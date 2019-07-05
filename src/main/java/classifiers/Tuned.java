package classifiers;

import classifiers.template_classifier.TemplateClassifier;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class Tuned extends TemplateClassifier {
    private final Supplier<AbstractClassifier> supplier;
    private final Function<Instances, ParameterSpace> parameterSpaceGetter;
    private Comparator<ClassifierResults> comparator = Comparator.comparingDouble(ClassifierResults::getAcc);
    private AbstractClassifier classifier;

    public Tuned(Supplier<AbstractClassifier> supplier, Function<Instances, ParameterSpace> parameterSpaceGetter) {
        this.supplier = supplier;
        this.parameterSpaceGetter = parameterSpaceGetter;
    }

    private static class ParameterBenchmark {
        public AbstractClassifier getClassifier() {
            return classifier;
        }

        private final AbstractClassifier classifier;
        private final ParameterSet parameterSet;

        public ParameterSet getParameterSet() {
            return parameterSet;
        }

        public ClassifierResults getTrainResults() {
            return trainResults;
        }

        private final ClassifierResults trainResults;

        private ParameterBenchmark(AbstractClassifier classifier, ParameterSet parameterSet, ClassifierResults trainResults) {
            this.classifier = classifier;
            this.parameterSet = parameterSet;
            this.trainResults = trainResults;
        }
    }

    private ClassifierResults evaluateParameter(AbstractClassifier classifier, ParameterSet parameterSet, Instances trainSet) throws Exception {
        classifier.setOptions(parameterSet.getOptions());
        ClassifierResults trainResults;
        if(classifier instanceof TemplateClassifier) {
            classifier.buildClassifier(trainSet);
            trainResults = ((TemplateClassifier) classifier).getTrainResults();
        } else {
            throw new UnsupportedOperationException();
        }
        return trainResults;
    }

    @Override
    public void buildClassifier(Instances trainSet) throws Exception {
        ParameterSpace parameterSpace = parameterSpaceGetter.apply(trainSet);
        parameterSpace.removeDuplicateValues();
        List<ParameterBenchmark> bestParameters = new ArrayList<>();
        AbstractClassifier classifier = supplier.get();
        ParameterSet parameterSet = parameterSpace.get(0);
        ClassifierResults trainResults = evaluateParameter(classifier, parameterSet, trainSet);
        bestParameters.add(new ParameterBenchmark(classifier, parameterSet, trainResults));
        for(int parameterSetIndex = 1; parameterSetIndex < parameterSpace.size(); parameterSetIndex++) {
            classifier = supplier.get();
            parameterSet = parameterSpace.get(parameterSetIndex);
            classifier.setOptions(parameterSet.getOptions());
            trainResults = evaluateParameter(classifier, parameterSet, trainSet);
            int comparison = comparator.compare(bestParameters.get(0).getTrainResults(), trainResults);
            if(comparison >= 0) {
                if(comparison > 0) {
                    bestParameters.clear();
                }
                bestParameters.add(new ParameterBenchmark(classifier, parameterSet, trainResults));
            }
        }
        this.classifier = bestParameters.get(getTrainRandom().nextInt(parameterSet.size())).getClassifier();
    }

    @Override
    public double[] distributionForInstance(Instance testInstance) throws Exception {
        return classifier.distributionForInstance(testInstance);
    }
}
