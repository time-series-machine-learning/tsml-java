
package evaluation.tuning;

import evaluation.ClassifierResults;
import evaluation.ClassifierResultsAnalysis;
import evaluation.tuning.evaluators.CrossValidationEvaluator;
import evaluation.tuning.evaluators.Evaluator;
import evaluation.tuning.searchers.GridSearcher;
import evaluation.tuning.searchers.ParameterSearcher;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class Tuner {
    private int seed = 0;
    private ParameterSearcher searcher = new GridSearcher();
    private Evaluator evaluator = new CrossValidationEvaluator();
    private Function<ClassifierResults, Double> evalMetric = ClassifierResults.GETTER_Accuracy;
       
    //for reporting purposes in results files/loggers
    public List<ParameterScorePair> allParasAndScores;

    
    /**
     * if true, the base classifier will be cloned in order to evaluate each parameter set 
     * this will prevent potentially any un-handled changes to the classifiers' state after 
     * the previous build/eval affecting the next one. 
     * 
     * if you know that the classifier either has no or correctly re-instantiates any data 
     * that would effect consecutive builds on the same classifier instance, just leave this as 
     * false to save mem/time
     */
    boolean cloneClassifierForEachParameterEval = false;
    
    public Tuner() { 
        
    }

    
    public boolean getCloneClassifierForEachParameterEval() {
        return cloneClassifierForEachParameterEval;
    }

    public void setCloneClassifierForEachParameterEval(boolean cloneClassifierForEachParameterEval) {
        this.cloneClassifierForEachParameterEval = cloneClassifierForEachParameterEval;
    }
    
    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

    public ParameterSearcher getSearcher() {
        return searcher;
    }

    public void setSearcher(ParameterSearcher searcher) {
        this.searcher = searcher;
    }

    public Evaluator getEvaluator() {
        return evaluator;
    }

    public void setEvaluator(Evaluator evaluator) {
        this.evaluator = evaluator;
    }

    public Function<ClassifierResults, Double> getEvalMetric() {
        return evalMetric;
    }

    public void setEvalMetric(Function<ClassifierResults, Double> evalMetric) {
        this.evalMetric = evalMetric;
    }
    
    
    public static class ParameterScorePair implements Comparable<ParameterScorePair>{ 
        public ParameterSet paras;
        public double score;

        public ParameterScorePair(ParameterSet parameterSet, double score) {
            this.paras = parameterSet;
            this.score = score;
        }

        @Override
        public int compareTo(ParameterScorePair other) {
            return Double.compare(this.score, other.score);
        }
    }
    
    public ParameterSet tune(AbstractClassifier baseClassifier, Instances trainSet, ParameterSpace parameterSpace) throws Exception {
        
//        System.out.println("Evaluating para space: " + parameterSpace);
        
        //init the space searcher
        searcher = new GridSearcher();
        searcher.setParameterSpace(parameterSpace);
        Iterator<ParameterSet> iter = searcher.iterator();
        
        //for reporting purposes in results files/loggers
        allParasAndScores = new ArrayList<>(parameterSpace.numUniqueParameterSets());
        
        //for resolving ties for the best paraset
        List<ParameterScorePair> tiesBestSoFar = new ArrayList<>();
        
        //iterate over the space
        while (iter.hasNext()) {
            ParameterSet pset = iter.next();
            String[] options = pset.toOptionsList();
            
            AbstractClassifier classifier = null;
            if (cloneClassifierForEachParameterEval) {
                //for some reason, the abstract classifiers' copy method returns a classifier interface reference...
                classifier = (AbstractClassifier)AbstractClassifier.makeCopy(baseClassifier); 
            }
            else {
                //just reuse the same instance of the classifier, assume that no info 
                //that from the previous build/eval affects this one.
                //potentially saves a lot of memory/time etc.
                classifier = baseClassifier;
            } 
                
            classifier.setOptions(options);
            
            ClassifierResults fullRes = evaluator.evaluate(classifier, trainSet);
            double score = evalMetric.apply(fullRes);
            
            ParameterScorePair paraScore = new ParameterScorePair(pset, score);
            allParasAndScores.add(paraScore);
            
            if (tiesBestSoFar.isEmpty()) //first time around loop
                tiesBestSoFar.add(paraScore);
            else {
                if (score == tiesBestSoFar.get(0).score) {
                    //another tie 
                    tiesBestSoFar.add(paraScore);
                }
                else if (score > tiesBestSoFar.get(0).score) {
                    //new best so far
                    tiesBestSoFar.clear();
                    tiesBestSoFar.add(paraScore);
                }
            } 
            
//            System.out.println("Score: " + String.format("%5f", score) + "\tParas: " + pset);
        }
        
        ParameterSet bestSet = null;
        if (tiesBestSoFar.size() == 1) {
            //clear winner
            bestSet = tiesBestSoFar.get(0).paras;
        }
        else { 
            //resolve ties randomly: todo future, maybe allow for some other method of resolving ties, 
            //e.g choose 'least complex' parameter set of the ties
            Random rand = new Random(seed);
            bestSet = tiesBestSoFar.get(rand.nextInt(tiesBestSoFar.size())).paras;
        }
             
//        System.out.println("Best parameter set was: " + bestSet);
        
        return bestSet;
    }
}
