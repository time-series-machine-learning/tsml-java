package utilities;

import java.io.FileNotFoundException;

/**
 * Essentially just a wrapper for all results of a single classifier over some set of datasets/folds,
 * with train and test splits. Functionality to read all results in, in a batch 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ClassifierResultsCollection implements DebugPrinting { 
    public ClassifierResults[][] trainResults = null;
    public ClassifierResults[][] testResults = null;

    public String[] datasets;
    public String classifierName;
    public int numFolds;

    public String baseReadPath;
    public boolean testResultsOnly;
    public boolean cleanResults;
    public boolean allowMissingResults;

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
    public ClassifierResultsCollection(String classifierName, String[] datasets, int numFolds, String baseReadPath, boolean testResultsOnly, boolean cleanResults, boolean allowMissingResults) throws Exception { 
        this.classifierName = classifierName;
        this.datasets = datasets;
        this.numFolds = numFolds;

        this.baseReadPath = baseReadPath;
        this.testResultsOnly = testResultsOnly;
        this.cleanResults = cleanResults;
        this.allowMissingResults = allowMissingResults;

        readInAllClassifierResults();
    }

    private void readInAllClassifierResults() throws Exception { 
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
