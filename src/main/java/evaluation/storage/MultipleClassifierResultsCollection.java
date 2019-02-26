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

import evaluation.storage.ClassifierResultsCollection;
import experiments.DataSets;
import evaluation.MultipleClassifierEvaluation;
import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.function.Function;
import statistics.tests.TwoSampleTests;
import utilities.DebugPrinting;
import utilities.ErrorReport;
import utilities.StatisticalUtilities;
import vector_classifiers.ChooseDatasetFromFile;

/**
 * Essentially wrapper for reading in/editing a 4d list of classifier results, implemented as a list of ClassifierResultsCollection
 * convertToBigOlArray() to get into the basic ClassifierResults[][][][] format
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class MultipleClassifierResultsCollection implements DebugPrinting { 
    public String[] datasets;
    public String[] classifierNames;
    public ClassifierResultsCollection[] allResults;
    public int numFolds;

    public String baseReadPath;
    public boolean testResultsOnly;
    public boolean cleanResults;
    public boolean allowMissingResults;

    /**
     * 
     * @param classifierNames
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
    public MultipleClassifierResultsCollection(String[] classifierNames, String[] datasets, int numFolds, String baseReadPath, boolean testResultsOnly, boolean cleanResults, boolean allowMissingResults) throws Exception { 
        this.classifierNames = classifierNames;
        this.datasets = datasets;
        this.numFolds = numFolds;

        this.baseReadPath = baseReadPath;
        this.testResultsOnly = testResultsOnly;
        this.cleanResults = cleanResults;
        this.allowMissingResults = allowMissingResults;

        this.allResults = new ClassifierResultsCollection[classifierNames.length];

        ErrorReport er = new ErrorReport("Results files not found:\n");
        for (int i = 0; i < classifierNames.length; i++) {
            try {
                allResults[i] = new ClassifierResultsCollection(classifierNames[i], datasets, numFolds, baseReadPath, testResultsOnly, cleanResults, allowMissingResults);
            } catch (Exception e) {
                er.log("Classifier Errors: " + classifierNames[i] + "\n" + e);
            }
        }

        if (allowMissingResults) {
            if (!er.isEmpty())
                printlnDebug(er.getLog());
        }
        else
            er.throwIfErrors();
    }

    /**
     * Train/test is outermost array to make it easier to ignore the null train results 
     * in the case that they aren't available, i.e 
     * 
     * ClassifierResults[][][][] allRes = mcrc.convertToBigOlArray();
     * ClassifierResults[][][] testOnly = allRes[1];
     * 
     * @return ClassifierResults[train/test][classifier][dataset][fold]
     */
    public ClassifierResults[][][][] convertToBigOlArray() {
        ClassifierResults[][][][] res = new ClassifierResults[2][allResults.length][][];

        for (int i = 0; i < classifierNames.length; i++) {
            res[0][i] = allResults[i].trainResults;
            res[1][i] = allResults[i].testResults;
        }

        return res;
    }

    /**
     * Classifiers will line up with the ordering of the classifierNames field
     * 
     * @return ClassifierResults[train/test][classifier][fold]
     */
    public ClassifierResults[][][] getAllResultsForDataset(String dataset) throws Exception { 
        ClassifierResults[][][] res = new ClassifierResults[2][allResults.length][];

        int dsetIndex = Arrays.asList(datasets).indexOf(dataset);
        if (dsetIndex == -1)
            throw new Exception("(getAllResultsForDataset) Results not loaded for dataset: " + dataset);

        for (int i = 0; i < classifierNames.length; i++) {
            res[0][i] = allResults[i].trainResults[dsetIndex];
            res[1][i] = allResults[i].testResults[dsetIndex];
        }

        return res;
    }

    /**
     * @return ClassifierResults[train/test][dataset][fold]
     */
    public ClassifierResults[][][] getAllResultsForClassifier(String classifierName) throws Exception { 
        int clsIndex = Arrays.asList(classifierNames).indexOf(classifierName);
        if (clsIndex == -1)
            throw new Exception("(getAllResultsForDataset) Results not loaded for dataset: " + classifierName);

        ClassifierResults[][][] res = new ClassifierResults[][][] { allResults[clsIndex].trainResults, allResults[clsIndex].testResults };
        return res;
    }


    public double[][][][] getAccuracies() {
        return getInfo((ClassifierResults cr) -> {return cr.getAcc();});
    }

    public double[][][][] getInfo(Function<ClassifierResults, Double> getter) {
        ClassifierResults[][][][] allRes = convertToBigOlArray();
        double[][][][] info = new double[2][classifierNames.length][datasets.length][numFolds];
        for (int i = testResultsOnly?1:0; i < 2; i++)
            for (int j = 0; j < classifierNames.length; j++)
                for (int k = 0; k < datasets.length; k++)
                    for (int l = 0; l < numFolds; l++) 
                        info[i][j][k][l] = getter.apply(allRes[i][j][k][l]);
        return info;
    }
    
    
}