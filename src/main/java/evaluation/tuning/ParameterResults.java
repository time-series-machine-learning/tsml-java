
package evaluation.tuning;

import evaluation.ClassifierResults;

/**
 * Simple container class for a parameter set and accompanying classifierResults, 
 * plus optionally a score which is used to order ParameterResults objects. 
 * 
 * Score defaults to the accuracy contained in the results object if not supplied
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ParameterResults implements Comparable<ParameterResults> { 
    public ParameterSet paras;
    public ClassifierResults results; 
    public double score;

    /**
     * Defaults to scoring by accuracy.
     */
    public ParameterResults(ParameterSet parameterSet, ClassifierResults results) {
        this.paras = parameterSet;
        this.results = results;
        this.score = results.acc;
    }
    
    public ParameterResults(ParameterSet parameterSet, ClassifierResults results, double score) {
        this.paras = parameterSet;
        this.results = results;
        this.score = score;
    }

    @Override
    public int compareTo(ParameterResults other) {
        return Double.compare(this.score, other.score);
    }
}
