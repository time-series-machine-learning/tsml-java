package tsml.classifiers.distance_based.utils.classifiers.results;

import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TrainEstimateTimeable;
import tsml.classifiers.TrainTimeable;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatchable;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Randomizable;

public class ResultUtils {

    public static void setNonResourceInfo(ClassifierResults results, final Classifier classifier, final Instances data) {
        results.setDatasetName(data.relationName());
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
    }

    public static void setInfo(ClassifierResults results, final Classifier classifier, final Instances data) {
        setNonResourceInfo(results, classifier, data);
        setMemoryInfo(results, classifier);
        setTimeInfo(results, classifier);
    }

    public static void setTimeInfo(ClassifierResults results, final Object obj) {
        if(obj instanceof TrainEstimateTimeable) {
            setTimeInfo(results, (TrainEstimateTimeable) obj);
        }
        if(obj instanceof TrainTimeable) {
            setTimeInfo(results, (TrainTimeable) obj);
        }
    }

    public static void setTimeInfo(ClassifierResults results, TrainTimeable obj) {
        results.setBuildTime(obj.getTrainTime());
        if(results.getErrorEstimateTime() < 0) {
            results.setErrorEstimateTime(0);
        }
        results.setBuildPlusEstimateTime(obj.getTrainTime() + results.getErrorEstimateTime());
        results.setTimeUnit(TimeUnit.NANOSECONDS);
    }

    public static void setTimeInfo(ClassifierResults results, TrainEstimateTimeable obj) {
        results.setErrorEstimateTime(obj.getTrainEstimateTime());
        results.setBuildPlusEstimateTime(obj.getTrainTime());
        results.setTimeUnit(TimeUnit.NANOSECONDS);
    }

    public static void setMemoryInfo(ClassifierResults results, final Object obj) {
        if(obj instanceof MemoryWatchable) {
            setMemoryInfo(results, (MemoryWatchable) obj);
        }
    }

    public static void setMemoryInfo(ClassifierResults results, final MemoryWatchable memoryWatchable) {
        results.setMemory(memoryWatchable.getMaxMemoryUsage());
    }

}
