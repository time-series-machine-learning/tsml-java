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
package evaluation.storage;

import java.io.FileNotFoundException;
import utilities.DebugPrinting;
import utilities.ErrorReport;

/**
 * Essentially a loader for many results over a given set of classifiers, datasets, folds, and splits
 * 
 * Assumes the 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class Collection_New implements DebugPrinting { 
    public ClassifierResults[][] trainResults = null;
    public ClassifierResults[][] testResults = null;

    public String[] datasets;
    public String classifierName;
    public int numFolds;

    public String baseReadPath;
    
    /**
     * if true, will null the individual prediction info of each ClassifierResults object after stats are found for it 
     * 
     * defaults to true
     */
    private boolean cleanResults = true;

    /**
     * if true, will not attempt to load trainFold results, and will not produce stats for train or traintestdiffs results
     * 
     * defaults to true
     */
    private boolean testResultsOnly = true;
    
    /**
     * if true, the returned lists are guaranteed to be of size numClassifiers*numDsets*numFolds*2,
     * but entries may be null;
     * 
     * defaults to false
     */
    public boolean allowMissingResults = false;

    /**
     * if true, will fill in missing probability distributions with one-hot vectors
     * for files read in that are missing them. intended for very old files, where you still 
     * want to calc auroc etc (metrics that need dists) for all the other classifiers 
     * that DO provide them, but also want to compare e.g accuracy with classifier that don't
     * 
     * defaults to false
     */
    private boolean ignoreMissingDistributions = false;
    
    
    
    
    
    
    
    /**
     * 
     * @param classifierName
     * @param datasets
     * @param numFolds
     * @param baseReadPath 
     * @param testResultsOnly if true, won't bother looking for/loading trainFold files
     * @param cleanResults if true, will delete the individual instance prediction info of each ClassifierResults object, 
     *      saving majority of mem requirements
     * @param allowMissingResults if false, missing fold files are reported and exception thrown, 
     *      else if true, will report results to debugprinting but just leave missing results as null.
     *      MOST other functionality here will not perform null checks. set to true mainly if just using 
     *      the batch file-reading functionality then doing your own stuff once loaded in
     * 
     * @throws Exception 
     */
    public Collection_New(String classifierName, String[] datasets, int numFolds, String baseReadPath, boolean testResultsOnly, boolean cleanResults, boolean allowMissingResults) throws Exception { 
        this.classifierName = classifierName;
        this.datasets = datasets;
        this.numFolds = numFolds;

        this.baseReadPath = baseReadPath;
        this.testResultsOnly = testResultsOnly;
        this.cleanResults = cleanResults;
        this.allowMissingResults = allowMissingResults;
    }
    
    /**
     * Loads all results files subject to the provided classifiers, datasets, folds and splits, 
     * and options set.
     */
    private void load() throws Exception { 
        if (baseReadPath.charAt(baseReadPath.length()-1) != '/')
            baseReadPath += "/";

        (new ClassifierResults()).printlnDebug(classifierName + " reading");

        int totalFnfs = 0;
        ErrorReport er = new ErrorReport("FileNotFoundExceptions thrown (### total):\n");

        testResults = new ClassifierResults[datasets.length][numFolds];
        if (testResultsOnly) 
            trainResults=null; //crappy but w/e
        else 
            trainResults = new ClassifierResults[datasets.length][numFolds];
        
        for (int d = 0; d < datasets.length; d++) {
            for (int f = 0; f < numFolds; f++) {

                if (!testResultsOnly) {
                    String trainFile = baseReadPath + classifierName + "/Predictions/" + datasets[d] + "/trainFold" + f + ".csv";
                    try {
                        trainResults[d][f] = new ClassifierResults(trainFile);
                        trainResults[d][f].findAllStatsOnce();
                        if (cleanResults)
                            trainResults[d][f].cleanPredictionInfo();
                    } catch (FileNotFoundException ex) {
                        er.log(trainFile + "\n");
                        totalFnfs++;
                        trainResults[d][f] = null;
                    }
                }

                String testFile = baseReadPath + classifierName + "/Predictions/" + datasets[d] + "/testFold" + f + ".csv";
                try {
                    testResults[d][f] = new ClassifierResults(testFile);
                    testResults[d][f].findAllStatsOnce();
                    if (cleanResults)
                        testResults[d][f].cleanPredictionInfo();
                } catch (FileNotFoundException ex) {
                    er.log(testFile + "\n");
                    totalFnfs++;
                    testResults[d][f] = null;
                }
            }
        }

        er.setLog(er.getLog().replace("###", totalFnfs+""));
        if (allowMissingResults)
            if (!er.isEmpty())
                printlnDebug(er.getLog());
        else
            er.throwIfErrors();

        (new ClassifierResults()).printlnDebug(classifierName + " successfully read in");
    }
}
