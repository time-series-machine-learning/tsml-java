package evaluation.evaluators;


import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import static utilities.GenericTools.indexOfMax;

import tsml.classifiers.Interpretable;
import tsml.classifiers.TSClassifier;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Simply gathers predictions from an already built/trained classifier on the data given
 *
 * As much meta info as possible shall be inferred (e.g. classifier name based on the class name),
 * but the calling function should explicitely set/check any meta info it wants to if accuracy is
 * important or the values non-standard (e.g. in this context you want the classifier name to
 * include some specific parameter identifier)
 *
 * distributionForInstance(Instance) MUST be defined, even if the classifier only really returns
 * a one-hot distribution
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class SingleTestSetEvaluatorTS {
    int seed;

    /**
     * Flag for whether to clone the data. Defaults to false, as no classifier should
     * be editing the data itself when training/testing, however setting this to true
     * will guarantee that the same (jave) instantiations of (weka) instance(s) objects
     * can be reused in higher-level experimental code.
     */
    boolean cloneData;

    /**
     * Each instance will have setClassMissing() called upon it. To ABSOLUTELY enforce that
     * no classifier can cheat in any way (e.g some filter/transform inadvertently incorporates the class
     * value back into the transformed data set).
     *
     * The only reason to leave this as false (as it has been by default, for backwards compatability reasons)
     * is that in higher level experimental code, the same (jave) instantiations of (weka) instance(s) objects are used multiple
     * times, and the latter expects the class value to still be there (to check for correct predictions, e.g)
     */
    boolean setClassMissing;

    public SingleTestSetEvaluatorTS(int seed, boolean cloneData, boolean setClassMissing) {
        this.seed = seed;
        this.cloneData = cloneData;
        this.setClassMissing = setClassMissing;
    }
    public SingleTestSetEvaluatorTS() {
        this(0,false,false);
    }


    private boolean vis = false;
    public SingleTestSetEvaluatorTS(int seed, boolean cloneData, boolean setClassMissing, boolean vis) {
        this(seed,cloneData,setClassMissing);
        this.vis = vis;
    }

    public synchronized ClassifierResults evaluate(TSClassifier classifier, TimeSeriesInstances dataset) throws Exception {


        ClassifierResults res = new ClassifierResults(dataset.numClasses());
        res.setTimeUnit(TimeUnit.NANOSECONDS);
        res.setClassifierName(classifier.toString());
        res.setDatasetName(dataset.getProblemName());
        res.setFoldID(seed);
        res.setSplit("train"); //todo revisit, or leave with the assumption that calling method will set this to test when needed

        res.turnOffZeroTimingsErrors();
        for (TimeSeriesInstance testinst : dataset) {
            double trueClassVal = testinst.getTargetValue();

            long startTime = System.nanoTime();
            double[] dist = classifier.distributionForInstance(testinst);
            long predTime = System.nanoTime() - startTime;

            if (vis) ((Interpretable)classifier).lastClassifiedInterpretability();

            res.addPrediction(trueClassVal, dist, indexOfMax(dist), predTime, ""); //todo indexOfMax does not break ties randomly.
        }

        res.turnOnZeroTimingsErrors();

        res.finaliseResults();
        res.findAllStatsOnce();

        return res;
    }

    /**
     * Utility method, will build on the classifier on the train set and evaluate on the test set
     */
    public synchronized ClassifierResults evaluate(TSClassifier classifier, TimeSeriesInstances train, TimeSeriesInstances test) throws Exception {
        long buildTime = System.nanoTime();
        classifier.buildClassifier(train);
        buildTime = System.nanoTime() - buildTime;

        ClassifierResults res = evaluate(classifier, test);

        res.turnOffZeroTimingsErrors();
        res.setBuildTime(buildTime);
        res.turnOnZeroTimingsErrors();

        return res;
    }

    public Evaluator cloneEvaluator() {
        return new SingleTestSetEvaluator(this.seed, this.cloneData, this.setClassMissing);
    }

}