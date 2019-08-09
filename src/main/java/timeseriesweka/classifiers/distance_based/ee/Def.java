package timeseriesweka.classifiers.distance_based.ee;

import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.TrainAccuracyEstimator;
import timeseriesweka.classifiers.distance_based.distances.DistanceMeasure;
import timeseriesweka.classifiers.distance_based.knn.Knn;
import utilities.ArrayUtilities;
import utilities.iteration.AbstractIterator;
import utilities.iteration.linear.LinearIterator;
import utilities.iteration.random.RandomIterator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

public class Def extends AbstractClassifier {

    private List<Function<Instances, ParameterSpace>> parameterSpaceFunctions = new ArrayList<>();
    private AbstractIterator<AbstractIterator<AbstractClassifier>> sourceIterator = new RandomIterator<>();
    private HashMap<String, Benchmark> constituents = new HashMap<>();

    private AbstractIterator<AbstractClassifier> generateIterator(ParameterSpace parameterSpace) throws Exception {
        AbstractIterator<AbstractClassifier> iterator = new RandomIterator<>();
        int size = parameterSpace.size();
        for(int i = 0; i < size; i++) {
            Knn knn = new Knn();
            ParameterSet parameterSet = parameterSpace.get(i);
            knn.setOptions(parameterSet.getOptions());
            knn.setTrainSize(2);
            iterator.add(knn);
        }
        return iterator;
    }

    private void populateParameterSpaces(Instances trainInstances) throws Exception {
        // populate parameter spaces
        for(Function<Instances, ParameterSpace> function : parameterSpaceFunctions) {
            ParameterSpace parameterSpace = function.apply(trainInstances);
            parameterSpace.removeDuplicateParameterSets();
            if(!parameterSpace.isEmpty()) {
                AbstractIterator<AbstractClassifier> iterator = generateIterator(parameterSpace);
                sourceIterator.add(iterator);
            }
        }
    }

    private String extractType(AbstractClassifier classifier) {
        String[] options = classifier.getOptions();
        String type = null;
        for(int i = 0; i < options.length; i += 2) {
            if(options[i].equals(DistanceMeasure.DISTANCE_MEASURE_KEY)) {
                type = options[i + 1];
                // todo if type == DTW || DDTW then need full / ed / warp
                break;
            }
        }
        return type;
    }

    @Override
    public void buildClassifier(Instances trainInstances) throws Exception {
        populateParameterSpaces(trainInstances);
        while (sourceIterator.hasNext()) {
            AbstractIterator<AbstractClassifier> iterator = sourceIterator.next();
            AbstractClassifier classifier = iterator.next();
            iterator.remove();
            ClassifierResults trainResults = evalClassifier(classifier, trainInstances);
            Benchmark benchmark = new Benchmark(classifier, trainResults);
            String type = extractType(classifier);
            Benchmark bestSoFar = constituents.get(type);
            if(bestSoFar.getTrainResults().getAcc() < benchmark.getTrainResults().getAcc()) {
                constituents.put(type, benchmark);
            }
            // todo feedback
            int trainSize = classifier.getTrainSize();
            if(trainSize + 1 <= trainInstances.size()) {
                classifier.setTrainSize(classifier.getTrainSize() + 1);
                iterator.add(classifier);
            }
            if(!iterator.hasNext()) {
                sourceIterator.remove();
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance testInstance) throws Exception {
        double[] distribution = new double[testInstance.numClasses()];
        for(Benchmark constituent : constituents.values()) {
            double weight = constituent.getTrainResults().getAcc();
            double[] constituentDistribution = constituent.getClassifier().distributionForInstance(testInstance);
            ArrayUtilities.multiplyInPlace(constituentDistribution, weight);
            ArrayUtilities.addInPlace(distribution, constituentDistribution);
        }
        ArrayUtilities.normaliseInPlace(distribution);
        return distribution;
    }

    private ClassifierResults evalClassifier(AbstractClassifier classifier, Instances trainInstances) throws Exception {
        ClassifierResults trainResults; // todo extract into tuner
        if(classifier instanceof TrainAccuracyEstimator) {
            classifier.buildClassifier(trainInstances);
            trainResults = ((TrainAccuracyEstimator) classifier).getTrainResults();
        } else {
            throw new UnsupportedOperationException();
        }
        return trainResults;
    }

}
