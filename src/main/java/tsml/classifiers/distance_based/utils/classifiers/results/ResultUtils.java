package tsml.classifiers.distance_based.utils.classifiers.results;

import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.TrainEstimateTimeable;
import tsml.classifiers.TrainTimeable;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatchable;
import tsml.data_containers.TimeSeriesInstances;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Randomizable;

public class ResultUtils {
    
    public static void setDataInfo(ClassifierResults results, Instances instances) {
        results.setDatasetName(instances.relationName());
    }
    
    public static void setDataInfo(ClassifierResults results, TimeSeriesInstances instances) {
        results.setDatasetName(instances.getProblemName());
    }
    
    public static void setClassifierInfo(ClassifierResults results, TSClassifier classifier) {
        // unpack wrapped tsclassifiers (e.g. an SMO wrapped as a TSClassifier). The TSClassifier doesn't implement whatever interfaces SMO does, e.g. Randomizable, therefore would not pickup the seed here for embedding in the results. So we unpack it to the raw classifier instance.
        // should the actual classifier be a TSClassifier, this will just return itself.
        setClassifierInfo(results, classifier.getClassifier());
    }
    
    public static void setClassifierInfo(ClassifierResults results, Classifier classifier) {
        if(classifier instanceof EnhancedAbstractClassifier) {
            results.setClassifierName(((EnhancedAbstractClassifier) classifier).getClassifierName());
            results.setFoldID(((EnhancedAbstractClassifier) classifier).getSeed());
            results.setParas(((EnhancedAbstractClassifier) classifier).getParameters());
            results.setErrorEstimateMethod(((EnhancedAbstractClassifier) classifier).getEstimatorMethod());
        } else {
            results.setClassifierName(classifier.getClass().getSimpleName());
            if(classifier instanceof OptionHandler) {
                results.setParas(StrUtils.join(",", ((OptionHandler) classifier).getOptions()));
            }
        }
        if(classifier instanceof Randomizable) {
            results.setFoldID(((Randomizable) classifier).getSeed());
        }
        if(classifier instanceof TrainTimeable) {
            results.setBuildTime(((TrainTimeable) classifier).getTrainTime());
            results.setErrorEstimateTime(0);
            results.setBuildPlusEstimateTime(results.getBuildTime());
            results.setTimeUnit(TimeUnit.NANOSECONDS);
        }
        if(classifier instanceof TrainEstimateTimeable) {
            results.setErrorEstimateTime(((TrainEstimateTimeable) classifier).getTrainEstimateTime());
            results.setBuildPlusEstimateTime(((TrainEstimateTimeable) classifier).getTrainPlusEstimateTime());
            results.setTimeUnit(TimeUnit.NANOSECONDS);
        }
        if(classifier instanceof MemoryWatchable) {
            results.setMemory(((MemoryWatchable) classifier).getMaxMemoryUsage());
        }
    }

    public static void setInfo(ClassifierResults results, final Classifier classifier, final Instances data) {
        setDataInfo(results, data);
        setClassifierInfo(results, classifier);
    }
    
    public static void setInfo(ClassifierResults results, TSClassifier classifier, TimeSeriesInstances data) {
        setDataInfo(results, data);
        setClassifierInfo(results, classifier);
    }
}
