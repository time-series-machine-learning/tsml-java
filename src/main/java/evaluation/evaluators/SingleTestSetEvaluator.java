/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package evaluation.evaluators;

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import static utilities.GenericTools.indexOfMax;

import tsml.classifiers.Interpretable;
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
public class SingleTestSetEvaluator extends Evaluator {
    
    public SingleTestSetEvaluator() {
        super(0,false,false);
    }
    public SingleTestSetEvaluator(int seed, boolean cloneData, boolean setClassMissing) {
        super(seed,cloneData,setClassMissing);
    }

    private boolean vis = false;
    public SingleTestSetEvaluator(int seed, boolean cloneData, boolean setClassMissing, boolean vis) {
        super(seed,cloneData,setClassMissing);
        this.vis = vis;
    }

    @Override
    public synchronized ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception {

        final Instances insts = cloneData ? new Instances(dataset) : dataset;

        ClassifierResults res = new ClassifierResults(insts.numClasses());
        res.setTimeUnit(TimeUnit.NANOSECONDS);
        res.setClassifierName(classifier.getClass().getSimpleName());
        res.setDatasetName(dataset.relationName());
        res.setFoldID(seed);
        res.setSplit("train"); //todo revisit, or leave with the assumption that calling method will set this to test when needed

        res.turnOffZeroTimingsErrors();
        for (Instance testinst : insts) {
            double trueClassVal = testinst.classValue();
            if (setClassMissing)
                testinst.setClassMissing();

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
    public synchronized ClassifierResults evaluate(Classifier classifier, Instances train, Instances test) throws Exception {
        long buildTime = System.nanoTime();
        classifier.buildClassifier(train);
        buildTime = System.nanoTime() - buildTime;
        
        ClassifierResults res = evaluate(classifier, test);
        
        res.turnOffZeroTimingsErrors();
        res.setBuildTime(buildTime);
        res.turnOnZeroTimingsErrors();
        
        return res;
    }

    @Override
    public Evaluator cloneEvaluator() {
        return new SingleTestSetEvaluator(this.seed, this.cloneData, this.setClassMissing);
    }
    
}
